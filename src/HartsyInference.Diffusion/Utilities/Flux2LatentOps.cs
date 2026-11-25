using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Latent layout shared by every family on the Flux.2 VAE (Flux.2, FLUX.3, Lens, IrisZ, MOSS): 2×2 patchify of the
/// 32-channel VAE latent to 128 channels, the VAE's BatchNorm running-stat normalization on those 128 channels, and the
/// <c>[B, C, H, W] ↔ [B, H·W, C]</c> token packing. Every op is a device op on the backend: the layouts are one
/// <see cref="IBackend.CopyStrided"/> over a permuted <see cref="StridedView"/>, and the BatchNorm is one
/// <see cref="IBackend.AffineBroadcastLastDim"/> over the packed tokens, whose last axis is the channel.</summary>
public static class Flux2LatentOps
{
    /// <summary>Packs <c>[B, C, H, W]</c> to <c>[B, H·W, C]</c> (diffusers <c>_pack_latents</c> with no spatial reshuffle).</summary>
    public static Tensor Pack(IBackend backend, Tensor latent)
    {
        int batch = (int)latent.Shape[0], channels = (int)latent.Shape[1], h = (int)latent.Shape[2], w = (int)latent.Shape[3];
        Tensor packed = new Tensor(new TensorShape(batch, h * w, channels), DType.F32);
        // Gather the view [B, H, W, C] of the NCHW latent: the output is row-major [B, H·W, C].
        backend.CopyStrided(packed, latent, StridedCopySpec.Gather(StridedView.Of(latent.Shape).Permute([0, 2, 3, 1])));
        return packed;
    }

    /// <summary>Unpacks <c>[B, H·W, C]</c> back to <c>[B, C, H, W]</c>.</summary>
    public static Tensor Unpack(IBackend backend, Tensor packed, int h, int w)
    {
        int batch = (int)packed.Shape[0], channels = (int)packed.Shape[2];
        Tensor unpacked = new Tensor(new TensorShape(batch, channels, h, w), DType.F32);
        backend.CopyStrided(unpacked, packed,
            StridedCopySpec.Gather(StridedView.Of(packed.Shape).Reshape([batch, h, w, channels]).Permute([0, 3, 1, 2])));
        return unpacked;
    }

    /// <summary>2×2 spatial patchify <c>[B, C, H·P, W·P] → [B, C·P², H, W]</c>; output channel <c>c·P² + py·P + px</c>.</summary>
    public static Tensor Patchify(IBackend backend, Tensor input, int outChannels, int patchSize)
    {
        int batch = (int)input.Shape[0], inChannels = (int)input.Shape[1], inH = (int)input.Shape[2], inW = (int)input.Shape[3];
        if (outChannels != inChannels * patchSize * patchSize)
        {
            throw new InvalidOperationException(
                $"Patchify: out_channels ({outChannels}) must equal in_channels * patch² ({inChannels} * {patchSize}²).");
        }
        int outH = inH / patchSize, outW = inW / patchSize;
        Tensor output = new Tensor(new TensorShape(batch, outChannels, outH, outW), DType.F32);
        // View the input as [B, C, H, py, W, px] and gather it as [B, C, py, px, H, W].
        StridedView source = StridedView.Of(input.Shape).Split(2, outH, patchSize).Split(4, outW, patchSize).Permute([0, 1, 3, 5, 2, 4]);
        backend.CopyStrided(output, input, StridedCopySpec.Gather(source));
        return output;
    }

    /// <summary>Inverse of <see cref="Patchify"/>: <c>[B, C·P², H, W] → [B, C, H·P, W·P]</c>.</summary>
    public static Tensor Unpatchify(IBackend backend, Tensor input, int outChannels, int patchSize)
    {
        int batch = (int)input.Shape[0], inChannels = (int)input.Shape[1], h = (int)input.Shape[2], w = (int)input.Shape[3];
        if (inChannels != outChannels * patchSize * patchSize)
        {
            throw new InvalidOperationException(
                $"Unpatchify: in_channels ({inChannels}) must equal out_channels * patch² ({outChannels} * {patchSize}²).");
        }
        Tensor output = new Tensor(new TensorShape(batch, outChannels, h * patchSize, w * patchSize), DType.F32);
        // The input is contiguous [B, C, P, P, H, W]; scatter it into the view [B, C, H, py, W, px] of the output.
        StridedView destination = StridedView.Of(output.Shape).Split(2, h, patchSize).Split(4, w, patchSize).Permute([0, 1, 3, 5, 2, 4]);
        backend.CopyStrided(output, input, StridedCopySpec.Scatter(destination));
        return output;
    }

    /// <summary>BatchNorm normalize per channel on NCHW: <c>(z − mean) / √(var + eps)</c>. Packs, runs the per-channel affine on
    /// the tokens, and unpacks; <paramref name="mean"/> and <paramref name="var"/> are F32 <c>[C]</c>.</summary>
    public static Tensor BnNormalize(IBackend backend, Tensor latent, Tensor mean, Tensor var, float eps)
        => BnNchw(backend, latent, mean, var, eps, unnormalize: false);

    /// <summary>BatchNorm un-normalize per channel on NCHW: <c>z·√(var + eps) + mean</c>.</summary>
    public static Tensor BnUnNormalize(IBackend backend, Tensor latent, Tensor mean, Tensor var, float eps)
        => BnNchw(backend, latent, mean, var, eps, unnormalize: true);

    /// <summary>BatchNorm normalize of packed tokens <c>[B, S, C]</c>: the per-channel affine
    /// <c>z·(1/√(var+eps)) − mean/√(var+eps)</c>, one broadcast over the last axis.</summary>
    public static Tensor BnNormalizeTokens(IBackend backend, Tensor packed, Tensor mean, Tensor var, float eps)
        => BnTokens(backend, packed, mean, var, eps, unnormalize: false);

    /// <summary>BatchNorm un-normalize of packed tokens <c>[B, S, C]</c>: <c>z·√(var+eps) + mean</c>, one broadcast over the last axis.</summary>
    public static Tensor BnUnNormalizeTokens(IBackend backend, Tensor packed, Tensor mean, Tensor var, float eps)
        => BnTokens(backend, packed, mean, var, eps, unnormalize: true);

    /// <summary>VAE latent <c>[B, C, H·P, W·P]</c> → BN-normalized packed tokens <c>[B, H·W, C·P²]</c>: one gather that patchifies
    /// and packs, then the per-channel affine.</summary>
    public static Tensor ToPackedTokens(IBackend backend, Tensor vaeLatent, Tensor mean, Tensor var, float eps, int patchSize)
    {
        int batch = (int)vaeLatent.Shape[0], channels = (int)vaeLatent.Shape[1];
        int h = (int)vaeLatent.Shape[2] / patchSize, w = (int)vaeLatent.Shape[3] / patchSize;
        Tensor tokens = new Tensor(new TensorShape(batch, h * w, channels * patchSize * patchSize), DType.F32);
        // [B, C, h, py, w, px] gathered as [B, h, w, C, py, px]: row-major [B, h·w, C·P²], channel c·P² + py·P + px.
        StridedView source = StridedView.Of(vaeLatent.Shape).Split(2, h, patchSize).Split(4, w, patchSize).Permute([0, 2, 4, 1, 3, 5]);
        backend.CopyStrided(tokens, vaeLatent, StridedCopySpec.Gather(source));
        Tensor normalized = BnTokens(backend, tokens, mean, var, eps, unnormalize: false);
        tokens.Dispose();
        return normalized;
    }

    /// <summary>Inverse of <see cref="ToPackedTokens"/>: packed tokens on an <c>h × w</c> grid → VAE latent <c>[B, C, h·P, w·P]</c>.</summary>
    public static Tensor FromPackedTokens(IBackend backend, Tensor packed, int h, int w, Tensor mean, Tensor var, float eps, int vaeChannels, int patchSize)
    {
        Tensor unnormalized = BnTokens(backend, packed, mean, var, eps, unnormalize: true);
        Tensor vae = new Tensor(new TensorShape(packed.Shape[0], vaeChannels, h * patchSize, w * patchSize), DType.F32);
        // The tokens are contiguous [B, h, w, C, P, P]; scatter them into the view [B, C, h, py, w, px] of the VAE latent.
        StridedView destination = StridedView.Of(vae.Shape).Split(2, h, patchSize).Split(4, w, patchSize).Permute([0, 2, 4, 1, 3, 5]);
        backend.CopyStrided(vae, unnormalized, StridedCopySpec.Scatter(destination));
        unnormalized.Dispose();
        return vae;
    }

    private static Tensor BnNchw(IBackend backend, Tensor latent, Tensor mean, Tensor var, float eps, bool unnormalize)
    {
        int h = (int)latent.Shape[2], w = (int)latent.Shape[3];
        using Tensor packed = Pack(backend, latent);
        using Tensor affine = BnTokens(backend, packed, mean, var, eps, unnormalize);
        return Unpack(backend, affine, h, w);
    }

    private static Tensor BnTokens(IBackend backend, Tensor packed, Tensor mean, Tensor var, float eps, bool unnormalize)
    {
        int channels = (int)packed.Shape[packed.Shape.Rank - 1];
        Tensor scale = new Tensor(new TensorShape(channels), DType.F32);
        Tensor shift = new Tensor(new TensorShape(channels), DType.F32);
        BnAffine(mean, var, eps, unnormalize, scale, shift);
        Tensor output = new Tensor(packed.Shape, DType.F32);
        backend.AffineBroadcastLastDim(output, packed, scale, shift);
        scale.Dispose();
        shift.Dispose();
        return output;
    }

    /// <summary>Fills the per-channel affine <c>out = in·scale + shift</c> of the BatchNorm: normalize uses
    /// <c>scale = 1/√(var+eps)</c>, <c>shift = −mean·scale</c>; un-normalize uses <c>scale = √(var+eps)</c>, <c>shift = mean</c>.</summary>
    private static unsafe void BnAffine(Tensor mean, Tensor var, float eps, bool unnormalize, Tensor scale, Tensor shift)
    {
        float* meanPtr = (float*)mean.DataPointer;
        float* varPtr = (float*)var.DataPointer;
        float* scalePtr = (float*)scale.DataPointer;
        float* shiftPtr = (float*)shift.DataPointer;
        for (int c = 0; c < scale.Shape[0]; c++)
        {
            float std = MathF.Sqrt(varPtr[c] + eps);
            if (unnormalize)
            {
                scalePtr[c] = std;
                shiftPtr[c] = meanPtr[c];
            }
            else
            {
                float inv = 1.0f / std;
                scalePtr[c] = inv;
                shiftPtr[c] = -meanPtr[c] * inv;
            }
        }
    }
}
