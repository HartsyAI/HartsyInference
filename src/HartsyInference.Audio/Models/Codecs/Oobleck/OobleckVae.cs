using HartsyInference.Core.Backends;
using HartsyInference.Core.Models;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.Oobleck;

/// <summary>Oobleck waveform VAE (Stability AI's Stable-Audio autoencoder; diffusers <c>AutoencoderOobleck</c>) — continuous 64-d latents at <c>sample_rate / hop</c> Hz ↔ stereo PCM.</summary>
/// <remarks>
/// Consumers: ACE-Step 1.5 (48 kHz, 25 Hz latents — <see cref="OobleckConfig.AceStep15"/>),
/// Stable Audio Open, and LTX-2's audio track.
///
/// <para>Weights load from the diffusers safetensors layout (<c>encoder.*</c> / <c>decoder.*</c>,
/// weight-normed convs as <c>weight_g</c>/<c>weight_v</c> fused at load, logscale snake params
/// exponentiated at load). Decode-only checkpoints (no <c>encoder.*</c> keys) are accepted —
/// <see cref="EncodeMode"/> then throws. <b>Numerics are validation-pending vs the Python
/// reference</b> (no weights in this environment); structure and shapes are CPU-tested.</para></remarks>
public sealed class OobleckVae : IAudioLatentDecoder, IAudioLatentEncoder
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

    /// <summary>Interface alias of <see cref="HasEncoder"/>.</summary>
    public bool CanEncode => _encoder is not null;

    /// <summary>PCM channel count (1 = mono, 2 = stereo).</summary>
    public int AudioChannels => _config.AudioChannels;

    /// <summary>Loads decoder (always) and encoder (when its keys are present) from a diffusers-layout weight dict.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _decoder.LoadWeights(w);
        if (w.ContainsKey("encoder.layers.0.weight_g"))
        {
            _encoder = new OobleckEncoder(_config, "encoder");
            _encoder.LoadWeights(w);
        }
    }

    /// <summary>Decodes latents <c>[B, latent_dim, T]</c> → PCM <c>[B, audio_channels, T · hop]</c> in (approximately) <c>[-1, 1]</c> — the decoder has no output tanh, so callers should clamp when converting to integer sample formats.</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        if (latent.Shape.Rank != 3 || (int)latent.Shape[1] != _config.DecoderInputChannels)
            throw new ArgumentException(
                $"Expected latent [B, {_config.DecoderInputChannels}, T]; got {latent.Shape}.", nameof(latent));
        return _decoder.Forward(backend, latent, (int)latent.Shape[0], (int)latent.Shape[2]);
    }

    /// <summary>Frames of latent <see cref="DecodeTiled"/> finishes per tile, and the context it keeps either side.
    /// Both match the release's own <c>decode_core_frames</c> / <c>decode_halo_frames</c>.</summary>
    public const int DefaultCoreFrames = 1024;

    /// <summary><inheritdoc cref="DefaultCoreFrames" path="/summary"/></summary>
    public const int DefaultHaloFrames = 16;

    /// <summary>Decodes in bounded tiles — same result as <see cref="Decode"/>, but peak activation memory is set by
    /// the tile rather than by the song.</summary>
    /// <remarks><para>The halo is what makes each tile's core <b>exact</b> rather than blended: every output sample
    /// in a core has its whole input support inside that tile, so there is no crossfade, no boundary smoothing and
    /// no seam — tiled and whole-song decodes agree sample for sample. That is also why this is safe to make the
    /// default: it changes memory, never numerics.</para>
    /// <para>Alignment is exact and independent of padding: a transpose conv maps input <c>i</c> to outputs
    /// <c>i·s + j − p</c>, so a tile starting at frame <c>left</c> produces output whose sample 0 is the whole
    /// song's sample <c>left · HopLength</c>, and that composes through the stack. Lengths are <b>not</b> a plain
    /// multiple though — see <see cref="OobleckConfig.DecodedLength"/>, which YuE2's odd stride makes 64 samples
    /// short — so the last tile ends where the whole decode ends rather than at a round boundary.</para></remarks>
    public unsafe Tensor DecodeTiled(IBackend backend, Tensor latent, int coreFrames = DefaultCoreFrames,
        int haloFrames = DefaultHaloFrames)
    {
        if (latent.Shape.Rank != 3 || (int)latent.Shape[1] != _config.DecoderInputChannels)
            throw new ArgumentException(
                $"Expected latent [B, {_config.DecoderInputChannels}, T]; got {latent.Shape}.", nameof(latent));
        if (coreFrames < 1)
            throw new ArgumentOutOfRangeException(nameof(coreFrames), coreFrames, "A tile core must be at least one frame.");
        if (haloFrames < 0)
            throw new ArgumentOutOfRangeException(nameof(haloFrames), haloFrames, "A tile halo cannot be negative.");

        int batch = (int)latent.Shape[0], latentDim = _config.DecoderInputChannels, frames = (int)latent.Shape[2];
        // A single tile would cover everything; decode straight through rather than paying for two extra copies.
        if (frames <= coreFrames) return Decode(backend, latent);

        int hop = _config.HopLength, channels = _config.AudioChannels;
        int total = checked((int)_config.DecodedLength(frames));
        Tensor audio = new(new TensorShape(batch, channels, total), DType.F32);
        try
        {
            float* destination = (float*)audio.DataPointer;
            float* source = (float*)latent.DataPointer;
            for (int start = 0; start < frames; start += coreFrames)
            {
                int end = Math.Min(frames, start + coreFrames);
                int left = Math.Max(0, start - haloFrames), right = Math.Min(frames, end + haloFrames);
                int span = right - left;

                using Tensor tile = new(new TensorShape(batch, latentDim, span), DType.F32);
                float* tileData = (float*)tile.DataPointer;
                for (int row = 0; row < batch * latentDim; row++)
                {
                    Buffer.MemoryCopy(source + (long)row * frames + left, tileData + (long)row * span,
                        (long)span * sizeof(float), (long)span * sizeof(float));
                }

                using Tensor decoded = Decode(backend, tile);
                int tileSamples = (int)decoded.Shape[2];
                // The final core runs to the end of the song, which the stack's edge loss leaves short of end·hop.
                int outStart = start * hop, outEnd = Math.Min(end * hop, total);
                int coreSamples = outEnd - outStart, cropStart = (start - left) * hop;
                if (cropStart + coreSamples > tileSamples)
                {
                    throw new InvalidOperationException(
                        $"An Oobleck tile of {span} frames decoded to {tileSamples} samples, too few to cover its "
                        + $"{coreSamples}-sample core at offset {cropStart}. Raise the halo.");
                }
                float* decodedData = (float*)decoded.DataPointer;
                for (int row = 0; row < batch * channels; row++)
                {
                    Buffer.MemoryCopy(decodedData + (long)row * tileSamples + cropStart,
                        destination + (long)row * total + outStart,
                        (long)coreSamples * sizeof(float), (long)coreSamples * sizeof(float));
                }
            }
            return audio;
        }
        catch
        {
            audio.Dispose();
            throw;
        }
    }

    /// <summary>Encodes PCM <c>[B, audio_channels, T]</c> (T a multiple of <see cref="HopLength"/>) to the deterministic latent mean <c>[B, latent_dim, T / hop]</c> — diffusers' <c>latent_dist.mode()</c>, the convention ACE-Step and Stable Audio use for conditioning.</summary>
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
