using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>The image autoencoder Paint runs its views through: pixels in <c>[-1, 1]</c> to scaled latents and back.</summary>
public interface IPaintLatentCodec
{
    /// <summary>Encodes <c>[N, 3, S, S]</c> images in <c>[-1, 1]</c> to scaled latents <c>[N, 4, S/8, S/8]</c>; the caller owns the result.</summary>
    Tensor Encode(IBackend backend, Tensor image);

    /// <summary>Decodes scaled latents <c>[N, 4, h, w]</c> to images <c>[N, 3, 8h, 8w]</c> in about <c>[-1, 1]</c>; the caller owns the result.</summary>
    Tensor Decode(IBackend backend, Tensor latents);
}
