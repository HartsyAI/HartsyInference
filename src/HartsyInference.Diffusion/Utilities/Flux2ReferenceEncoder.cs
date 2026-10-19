using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Turns reference images into FLUX.2-latent reference tokens: VAE-encode, BatchNorm-normalize and 2×2-pack,
/// the same treatment the target latent gets, so a DiT can embed them with its own image embedder.</summary>
public static class Flux2ReferenceEncoder
{
    /// <summary>Encodes each <c>[1, 3, H, W]</c> image in <c>[-1, 1]</c> (sides multiples of
    /// <paramref name="downscaleFactor"/>) and returns its host-resident tokens, in order. The caller disposes them.</summary>
    public static unsafe ReferenceLatentTokens[] Encode(IBackend backend, VaeEncoder encoder, IReadOnlyList<Tensor> images, Tensor bnMean,
        Tensor bnVar, float bnEps, int patchSize, int downscaleFactor, string family)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(images);
        ReferenceLatentTokens[] encoded = new ReferenceLatentTokens[images.Count];
        int done = 0;
        try
        {
            foreach (Tensor image in images)
            {
                int h = (int)image.Shape[2], w = (int)image.Shape[3];
                if (h % downscaleFactor != 0 || w % downscaleFactor != 0)
                    throw new ArgumentException($"{family} reference images need sides that are multiples of {downscaleFactor}; got {w}x{h}.");
                using Tensor vaeLatent = encoder.Encode(backend, image);
                Tensor tokens = Flux2LatentOps.ToPackedTokens(vaeLatent, bnMean, bnVar, bnEps, patchSize);
                // Host copy, so the VAE phase's activation sweep cannot take it.
                _ = tokens.DataPointer;
                encoded[done++] = new ReferenceLatentTokens(tokens, h / downscaleFactor, w / downscaleFactor);
            }
            return encoded;
        }
        catch
        {
            for (int i = 0; i < done; i++) encoded[i].Tokens.Dispose();
            throw;
        }
    }
}
