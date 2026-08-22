using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VibeVoice;

/// <summary>Semantic VAE — an encoder-only twin of the acoustic VAE, used to re-encode each decoded audio frame back into the LM's input space with <c>vae_dim = 128</c> and deterministic sampling (no posterior randomness).</summary>
/// <remarks>Used in the multi-speaker (non-streaming) variant only. The streaming-0.5B model
/// has no semantic VAE; per-frame feedback is acoustic-only.</remarks>
internal sealed class VibeVoiceSemanticTokenizerModel
{
    private readonly VibeVoiceTokenizerConfig _config;
    private readonly VibeVoiceTokenizerEncoder _encoder;

    public VibeVoiceTokenizerConfig Config => _config;

    public VibeVoiceSemanticTokenizerModel(VibeVoiceTokenizerConfig config, string prefix)
    {
        if (config.StdDistType != "none")
            throw new ArgumentException($"Semantic VAE expects std_dist_type='none', got '{config.StdDistType}'.", nameof(config));
        _config = config;
        _encoder = new VibeVoiceTokenizerEncoder(config, $"{prefix}.encoder");
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w) => _encoder.LoadWeights(w);

    /// <summary>Encodes a decoded audio chunk <c>[batch, 1, T_pcm]</c> into the deterministic-mean semantic latent <c>[batch, T_pcm / 3200, vae_dim=128]</c> (no sampling).</summary>
    public Tensor Encode(IBackend backend, Tensor pcm, int batch, int tPcm,
        VibeVoiceTokenizerStreamingCache? cache = null, ReadOnlySpan<int> sampleIndices = default)
    {
        Tensor latentCf = _encoder.Forward(backend, pcm, batch, tPcm, cache, sampleIndices);
        int tLatent = (int)latentCf.Shape[2];

        Tensor mean = new(new TensorShape(batch, tLatent, _config.VaeDim), DType.F32);
        backend.Transpose2D(mean, latentCf, _config.VaeDim, tLatent);
        latentCf.Dispose();
        return mean;
    }

    public IEnumerable<Tensor> EnumerateWeights() => _encoder.EnumerateWeights();
}
