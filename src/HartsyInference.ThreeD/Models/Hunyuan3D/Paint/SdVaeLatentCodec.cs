using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>The Stable Diffusion 2 <c>AutoencoderKL</c> (scale 0.18215) through the engine's VAE. The encoder returns the posterior mean where upstream samples it.</summary>
public sealed class SdVaeLatentCodec : IPaintLatentCodec
{
    private readonly VaeEncoder _encoder;
    private readonly VaeDecoder _decoder;

    /// <summary>Wraps loaded VAE halves.</summary>
    public SdVaeLatentCodec(VaeEncoder encoder, VaeDecoder decoder)
    {
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _decoder = decoder ?? throw new ArgumentNullException(nameof(decoder));
    }

    /// <inheritdoc/>
    public Tensor Encode(IBackend backend, Tensor image) => _encoder.Encode(backend, image);

    /// <inheritdoc/>
    public Tensor Decode(IBackend backend, Tensor latents) => _decoder.Decode(backend, latents);
}
