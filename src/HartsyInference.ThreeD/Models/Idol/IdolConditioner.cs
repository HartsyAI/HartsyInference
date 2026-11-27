using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ThreeD.Preprocessing;

namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>Idol's two image streams (idol-sdk's <c>ImageEncoder</c>) for a prepared 1024 px picture: <c>feature1</c>, the DINOv3 ViT-H+
/// tokens after the model's final norm and a non-affine LayerNorm <c>[1, 4101, 1280]</c>, and <c>feature2</c>, the FLUX.2 VAE latent
/// (the posterior mean) patchified 2 × 2, BatchNorm-normalised, flattened to 4096 tokens of 128 channels and prefixed by five zero tokens
/// <c>[1, 4101, 128]</c>. Idol samples the posterior; Budget3D's encoder, and this one, take its mean. Stateless: each stage is run on its own so the encoders need not be loaded together.</summary>
public static class IdolConditioner
{
    /// <summary><c>batch_norm_eps</c> of the FLUX.2 VAE's config.</summary>
    public const float BatchNormEps = 1e-4f;

    private const int PatchSize = 2;
    private const int ZeroTokens = 5;

    /// <summary>The DINOv3 stream <c>[1, 4101, 1280]</c> of the prepared <c>1024 × 1024</c> RGB24 picture from a ViT-H+/16 conditioner loaded for 1024 px; the caller owns it.</summary>
    public static Tensor DinoTokens(IBackend backend, Dinov3ImageConditioner dino, ReadOnlySpan<byte> rgb)
    {
        ArgumentNullException.ThrowIfNull(dino);
        Require(rgb);
        return dino.Encode(backend, rgb, IdolImageProcessor.Size, IdolImageProcessor.Size, IdolImageProcessor.Size, finalNorm: true);
    }

    /// <summary>The VAE stream of the prepared <c>1024 × 1024</c> RGB24 picture from an encoder and the VAE's BatchNorm statistics; the
    /// caller owns the result.</summary>
    public static unsafe Tensor VaeTokens(IBackend backend, VaeEncoder vae, Tensor bnMean, Tensor bnVar, ReadOnlySpan<byte> rgb)
    {
        Require(rgb);
        int side = IdolImageProcessor.Size, plane = side * side;
        using Tensor pixels = new(new TensorShape(1, 3, side, side), DType.F32);
        float* p = (float*)pixels.DataPointer;
        for (int i = 0; i < plane; i++)
        {
            for (int c = 0; c < 3; c++)
                p[c * plane + i] = rgb[i * 3 + c] / 255f * 2f - 1f;
        }

        using Tensor latent = vae.Encode(backend, pixels);
        using Tensor tokens = Flux2LatentOps.ToPackedTokens(backend, latent, bnMean, bnVar, BatchNormEps, PatchSize);
        using Tensor zeros = new(new TensorShape(1, ZeroTokens, tokens.Shape[2]), DType.F32);
        Tensor result = new(new TensorShape(1, ZeroTokens + tokens.Shape[1], tokens.Shape[2]), DType.F32);
        backend.Concat(result, [zeros, tokens], 1);
        return result;
    }

    private static void Require(ReadOnlySpan<byte> rgb)
    {
        int expected = IdolImageProcessor.Size * IdolImageProcessor.Size * 3;
        if (rgb.Length != expected)
            throw new ArgumentException($"Idol's encoders take a prepared {IdolImageProcessor.Size} px RGB24 picture ({expected} bytes); got {rgb.Length}.", nameof(rgb));
    }
}
