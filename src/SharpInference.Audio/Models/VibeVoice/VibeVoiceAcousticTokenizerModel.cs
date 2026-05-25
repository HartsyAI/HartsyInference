using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.VibeVoice;

/// <summary>Acoustic VAE — the encoder + decoder pair that turns 24 kHz mono PCM into
/// 64-d latents at 7.5 Hz and back. Mirrors the Python
/// <c>VibeVoiceAcousticTokenizerModel</c>: thin glue around
/// <see cref="VibeVoiceTokenizerEncoder"/> and <see cref="VibeVoiceTokenizerDecoder"/> with
/// Gaussian sampling on the encoder output.
///
/// <para>Sampling modes (per <c>std_dist_type</c>):
/// <list type="bullet">
///   <item><c>"fix"</c> — <c>z = mean + fix_std * N(0, I)</c> with a fixed scalar std.</item>
///   <item><c>"gaussian"</c> (default) — per-batch random std multiplier
///         <c>std_b = N(0, fix_std / 0.8)</c> then <c>z = mean + std_b * N(0, I)</c>.</item>
///   <item><c>"none"</c> — deterministic, returns the mean. Used by the semantic VAE.</item>
/// </list></para>
///
/// <para>Output of <see cref="Encode"/> is the channel-last <c>[B, T, vae_dim]</c> mean
/// tensor (the Python <c>VibeVoiceTokenizerEncoderOutput.mean</c>). Call
/// <see cref="Sample"/> to draw the actual latent; <see cref="Decode"/> takes the latent
/// back to PCM.</para></summary>
internal sealed unsafe class VibeVoiceAcousticTokenizerModel
{
    private readonly VibeVoiceTokenizerConfig _config;
    private readonly VibeVoiceTokenizerEncoder _encoder;
    private readonly VibeVoiceTokenizerDecoder _decoder;

    public VibeVoiceTokenizerConfig Config => _config;
    public float FixStd => _config.FixStd;
    public string StdDistType => _config.StdDistType;

    public VibeVoiceAcousticTokenizerModel(VibeVoiceTokenizerConfig config, string prefix)
    {
        _config = config;
        _encoder = new VibeVoiceTokenizerEncoder(config, $"{prefix}.encoder");
        _decoder = new VibeVoiceTokenizerDecoder(config, $"{prefix}.decoder");
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _encoder.LoadWeights(w);
        _decoder.LoadWeights(w);
    }

    /// <summary>Encodes a PCM waveform into a channel-last latent mean tensor.
    /// Input <c>[batch, 1, T_pcm]</c>, output <c>[batch, T_pcm / 3200, vae_dim]</c>.
    /// Streaming when <paramref name="cache"/> is non-null.</summary>
    public Tensor Encode(IBackend backend, Tensor pcm, int batch, int tPcm,
        VibeVoiceTokenizerStreamingCache? cache = null, ReadOnlySpan<int> sampleIndices = default)
    {
        // Encoder runs in channels-first and returns [batch, vae_dim, T_latent].
        Tensor latentCf = _encoder.Forward(backend, pcm, batch, tPcm, cache, sampleIndices);
        int tLatent = (int)latentCf.Shape[2];

        // Permute to channels-last [batch, T_latent, vae_dim] — matches Python's
        // `latents.permute(0, 2, 1)` in `encode()`.
        Tensor mean = new(new TensorShape(batch, tLatent, _config.VaeDim), DType.F32);
        backend.Transpose2D(mean, latentCf, _config.VaeDim, tLatent);
        latentCf.Dispose();
        return mean;
    }

    /// <summary>Samples from the VAE posterior. <paramref name="mean"/> is
    /// channels-last <c>[batch, T, vae_dim]</c>. Returns a fresh tensor of the same
    /// shape; caller owns disposal.
    ///
    /// <para>For <c>"none"</c> distType, this just clones the mean — the semantic VAE
    /// case (encoder-only, deterministic).</para></summary>
    public Tensor Sample(Tensor mean, Random rng, string? distTypeOverride = null)
    {
        string mode = distTypeOverride ?? _config.StdDistType;
        Tensor result = new(mean.Shape, DType.F32);
        long n = mean.ElementCount;
        float* src = (float*)mean.DataPointer;
        float* dst = (float*)result.DataPointer;

        switch (mode)
        {
            case "none":
                for (long i = 0; i < n; i++) dst[i] = src[i];
                return result;

            case "fix":
                {
                    float fixStd = _config.FixStd;
                    for (long i = 0; i < n; i++) dst[i] = src[i] + fixStd * SampleStandardNormal(rng);
                    return result;
                }

            case "gaussian":
                {
                    // Per-batch random std multiplier: std_b = N(0, fix_std/0.8). Apply same
                    // scalar to every (T, vae_dim) entry of that batch element.
                    int batch = (int)mean.Shape[0];
                    int t = (int)mean.Shape[1];
                    int d = (int)mean.Shape[2];
                    float stdScale = _config.FixStd / 0.8f;
                    for (int b = 0; b < batch; b++)
                    {
                        float stdB = SampleStandardNormal(rng) * stdScale;
                        for (int j = 0; j < t; j++)
                        {
                            int rowBase = (b * t + j) * d;
                            for (int k = 0; k < d; k++)
                                dst[rowBase + k] = src[rowBase + k] + stdB * SampleStandardNormal(rng);
                        }
                    }
                    return result;
                }

            default:
                result.Dispose();
                throw new ArgumentException($"Unsupported std_dist_type '{mode}'.", nameof(distTypeOverride));
        }
    }

    /// <summary>Decodes a channels-last latent back to PCM. Input
    /// <c>[batch, T_latent, vae_dim]</c>; if the input is already channels-first
    /// <c>[batch, vae_dim, T_latent]</c> (the Python code checks
    /// <c>latents.shape[1] == config.vae_dim</c>) it's used directly.
    ///
    /// <para>Output is <c>[batch, 1, T_latent * 3200]</c>.</para></summary>
    public Tensor Decode(IBackend backend, Tensor latent, int batch,
        VibeVoiceTokenizerStreamingCache? cache = null, ReadOnlySpan<int> sampleIndices = default)
    {
        // Accept both channels-last [B, T, D] and channels-first [B, D, T].
        int dim1 = (int)latent.Shape[1];
        Tensor latentCf;
        int tLatent;
        bool ownsCf;
        if (dim1 == _config.VaeDim)
        {
            latentCf = latent;
            ownsCf = false;
            tLatent = (int)latent.Shape[2];
        }
        else
        {
            tLatent = (int)latent.Shape[1];
            latentCf = new(new TensorShape(batch, _config.VaeDim, tLatent), DType.F32);
            backend.Transpose2D(latentCf, latent, tLatent, _config.VaeDim);
            ownsCf = true;
        }

        Tensor pcm = _decoder.Forward(backend, latentCf, batch, tLatent, cache, sampleIndices);
        if (ownsCf) latentCf.Dispose();
        return pcm;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _encoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
    }

    /// <summary>Box-Muller draw of one standard normal sample. Used inside the per-element
    /// VAE sampling loop. Not the fastest option but matches Python's
    /// <c>torch.randn_like</c> distribution shape and is fully deterministic given the
    /// caller's <see cref="Random"/>.</summary>
    private static float SampleStandardNormal(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}
