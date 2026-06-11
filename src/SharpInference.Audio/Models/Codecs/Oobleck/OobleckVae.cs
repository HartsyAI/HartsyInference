using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Codecs.Oobleck;

/// <summary>Oobleck waveform VAE (Stability AI's Stable-Audio autoencoder; diffusers
/// <c>AutoencoderOobleck</c>). Continuous 64-d latents at <c>sample_rate / hop</c> Hz ↔ stereo PCM.
/// Consumers: ACE-Step 1.5 (48 kHz, 25 Hz latents — <see cref="OobleckConfig.AceStep15"/>),
/// Stable Audio Open, and LTX-2's audio track.
///
/// <para>Weights load from the diffusers safetensors layout (<c>encoder.*</c> / <c>decoder.*</c>,
/// weight-normed convs as <c>weight_g</c>/<c>weight_v</c> fused at load, logscale snake params
/// exponentiated at load). Decode-only checkpoints (no <c>encoder.*</c> keys) are accepted —
/// <see cref="EncodeMode"/> then throws. <b>Numerics are validation-pending vs the Python
/// reference</b> (no weights in this environment); structure and shapes are CPU-tested.</para></summary>
public sealed class OobleckVae
{
    private readonly OobleckConfig _config;
    private readonly OobleckDecoder _decoder;
    private OobleckEncoder? _encoder;

    public OobleckVae(OobleckConfig config)
    {
        _config = config;
        _decoder = new OobleckDecoder(config, "decoder");
    }

    /// <summary>Latent dimensionality (decoder input channels).</summary>
    public int LatentDim => _config.DecoderInputChannels;

    /// <summary>Samples per latent frame.</summary>
    public int HopLength => _config.HopLength;

    /// <summary>True once encoder weights were found and loaded (needed for <see cref="EncodeMode"/>).</summary>
    public bool HasEncoder => _encoder is not null;

    /// <summary>Loads decoder (always) and encoder (when its keys are present) from a
    /// diffusers-layout weight dict.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _decoder.LoadWeights(w);
        if (w.ContainsKey("encoder.conv1.weight_g"))
        {
            _encoder = new OobleckEncoder(_config, "encoder");
            _encoder.LoadWeights(w);
        }
    }

    /// <summary>Decodes latents <c>[B, latent_dim, T]</c> → PCM <c>[B, audio_channels, T · hop]</c>
    /// in (approximately) <c>[-1, 1]</c> — the decoder has no output tanh, so callers should clamp
    /// when converting to integer sample formats.</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        if (latent.Shape.Rank != 3 || (int)latent.Shape[1] != _config.DecoderInputChannels)
            throw new ArgumentException(
                $"Expected latent [B, {_config.DecoderInputChannels}, T]; got {latent.Shape}.", nameof(latent));
        return _decoder.Forward(backend, latent, (int)latent.Shape[0], (int)latent.Shape[2]);
    }

    /// <summary>Encodes PCM <c>[B, audio_channels, T]</c> (T a multiple of <see cref="HopLength"/>)
    /// to the deterministic latent mean <c>[B, latent_dim, T / hop]</c> — diffusers'
    /// <c>latent_dist.mode()</c>, the convention ACE-Step and Stable Audio use for conditioning.</summary>
    public unsafe Tensor EncodeMode(IBackend backend, Tensor pcm)
    {
        if (_encoder is null)
            throw new InvalidOperationException("This Oobleck checkpoint has no encoder weights — decode-only.");
        if (pcm.Shape.Rank != 3 || (int)pcm.Shape[1] != _config.AudioChannels)
            throw new ArgumentException($"Expected PCM [B, {_config.AudioChannels}, T]; got {pcm.Shape}.", nameof(pcm));
        if (pcm.Shape[2] % _config.HopLength != 0)
            throw new ArgumentException($"PCM length {pcm.Shape[2]} must be a multiple of the hop ({_config.HopLength}).", nameof(pcm));

        Tensor parameters = _encoder.Forward(backend, pcm, (int)pcm.Shape[0], (int)pcm.Shape[2]);
        try
        {
            // parameters = [B, 2·latent_dim, T_lat]; the mean is the first half of the channels.
            int batch = (int)parameters.Shape[0];
            int half = _config.DecoderInputChannels;
            int tLat = (int)parameters.Shape[2];
            Tensor mean = new(new TensorShape(batch, half, tLat), DType.F32);
            float* src = (float*)parameters.DataPointer;
            float* dst = (float*)mean.DataPointer;
            long perBatchSrc = (long)2 * half * tLat;
            long perBatchDst = (long)half * tLat;
            for (int b = 0; b < batch; b++)
                Buffer.MemoryCopy(src + b * perBatchSrc, dst + b * perBatchDst, perBatchDst * 4, perBatchDst * 4);
            return mean;
        }
        finally
        {
            parameters.Dispose();
        }
    }

    /// <summary>Enumerates all loaded weights for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
        if (_encoder is not null)
            foreach (Tensor t in _encoder.EnumerateWeights()) yield return t;
    }
}
