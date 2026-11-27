using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>The seed-frame encode over a loaded LTX-2 VAE encoder: the posterior mode, then
/// <see cref="LtxVideo2VaeDecoder.NormalizeLatent"/> once with the checkpoint's <c>latents_mean</c> and <c>latents_std</c>
/// (scaling factor 1). Borrows the encoder; its owner releases it.</summary>
public sealed class SanaWmLtx2ImageEncoder : ISanaWmImageEncoder
{
    private readonly IBackend _backend;
    private readonly LtxVideo2VaeEncoder _encoder;
    private readonly float[] _latentsMean;
    private readonly float[] _latentsStd;

    /// <summary>Wraps a loaded encoder and the per-channel latent statistics.</summary>
    public SanaWmLtx2ImageEncoder(IBackend backend, LtxVideo2VaeEncoder encoder, float[] latentsMean, float[] latentsStd)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(latentsMean);
        ArgumentNullException.ThrowIfNull(latentsStd);
        if (latentsMean.Length != latentsStd.Length || latentsMean.Length == 0)
            throw new ArgumentException("Latent mean and std must be non-empty and the same length.", nameof(latentsStd));
        _backend = backend;
        _encoder = encoder;
        _latentsMean = latentsMean;
        _latentsStd = latentsStd;
    }

    /// <inheritdoc/>
    public Tensor Encode(Tensor video)
    {
        using Tensor mode = _encoder.Encode(_backend, video);
        return LtxVideo2VaeDecoder.NormalizeLatent(mode, _latentsMean, _latentsStd);
    }
}
