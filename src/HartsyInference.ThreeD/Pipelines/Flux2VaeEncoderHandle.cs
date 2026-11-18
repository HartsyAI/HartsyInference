using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>The FLUX.2 VAE as Idol's conditioner needs it: the encoder, the BatchNorm statistics of the patchified latent, and what to
/// dispose when the encoder is done. The engine builds it from its FLUX.2 VAE loader; this package cannot reference the engine.</summary>
public sealed class Flux2VaeEncoderHandle : IDisposable
{
    private readonly IDisposable? _owner;

    /// <param name="encoder">The VAE encoder.</param>
    /// <param name="bnMean">The BatchNorm running mean of the patchified latent, F32 <c>[128]</c>.</param>
    /// <param name="bnVar">The running variance, F32 <c>[128]</c>.</param>
    /// <param name="owner">What holds the encoder's weights, disposed with the handle; null when the caller owns them.</param>
    public Flux2VaeEncoderHandle(VaeEncoder encoder, Tensor bnMean, Tensor bnVar, IDisposable? owner = null)
    {
        Encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        BnMean = bnMean ?? throw new ArgumentNullException(nameof(bnMean));
        BnVar = bnVar ?? throw new ArgumentNullException(nameof(bnVar));
        _owner = owner;
    }

    /// <summary>The VAE encoder.</summary>
    public VaeEncoder Encoder { get; }

    /// <summary>The BatchNorm running mean of the patchified latent.</summary>
    public Tensor BnMean { get; }

    /// <summary>The BatchNorm running variance of the patchified latent.</summary>
    public Tensor BnVar { get; }

    /// <summary>Releases the owner of the encoder's weights.</summary>
    public void Dispose() => _owner?.Dispose();
}
