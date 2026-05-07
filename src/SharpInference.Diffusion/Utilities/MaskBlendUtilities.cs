using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Utilities;

/// <summary>Shared helpers for masked-inpaint blending used by every img2img-capable pipeline (SDXL, Flux, SD3, …). The math here is the same diffusers' <c>StableDiffusionInpaintPipelineLegacy</c> uses for "soft inpainting": at every scheduler step the unmasked region is replaced with a freshly re-noised copy of the source latent at the next step's sigma, while the masked region is left to denoise freely.
/// <para>Centralised here so each pipeline implementation gets the same boundary semantics — pipelines diverge in how they pack latents (Flux 2×2 patchify, Sd3 patch_size=2) but the underlying blend is identical and we want one canonical implementation to audit.</para></summary>
public static unsafe class MaskBlendUtilities
{
    /// <summary>Area-average downsample of a <c>[1, 1, H, W]</c> mask to <c>[1, 1, latentH, latentW]</c>. Each output cell is the mean over the corresponding source block. Requires H, W to be exact multiples of latent dims (true for VAE 8× compression on every supported model).</summary>
    public static Tensor DownsampleMaskAreaAverage(Tensor mask, int latentH, int latentW)
    {
        if (mask.Shape.Rank != 4 || mask.Shape[0] != 1 || mask.Shape[1] != 1)
        {
            throw new ArgumentException($"Mask must be [1, 1, H, W]; got {mask.Shape}.", nameof(mask));
        }
        if (mask.DType != DType.F32)
        {
            throw new ArgumentException($"Mask must be F32; got {mask.DType}.", nameof(mask));
        }
        int srcH = (int)mask.Shape[2];
        int srcW = (int)mask.Shape[3];
        int blockH = srcH / latentH;
        int blockW = srcW / latentW;
        if (blockH * latentH != srcH || blockW * latentW != srcW)
        {
            throw new ArgumentException(
                $"Mask dims ({srcH}x{srcW}) must be exact multiples of latent dims ({latentH}x{latentW}).");
        }
        float invBlock = 1f / (blockH * blockW);
        Tensor latentMask = new Tensor(new TensorShape(1, 1, latentH, latentW), DType.F32);
        float* sp = (float*)mask.DataPointer;
        float* dp = (float*)latentMask.DataPointer;
        for (int y = 0; y < latentH; y++)
        {
            for (int x = 0; x < latentW; x++)
            {
                float sum = 0f;
                int srcRow = y * blockH;
                int srcCol = x * blockW;
                for (int by = 0; by < blockH; by++)
                {
                    int rowOff = (srcRow + by) * srcW + srcCol;
                    for (int bx = 0; bx < blockW; bx++)
                    {
                        sum += sp[rowOff + bx];
                    }
                }
                dp[y * latentW + x] = sum * invBlock;
            }
        }
        return latentMask;
    }

    /// <summary>In-place mask blend for a <c>[1, C, H, W]</c> tensor against a same-shape source: <c>target = target * mask + source * (1 - mask)</c>. The mask is <c>[1, 1, H, W]</c> in <c>[0, 1]</c> (1 = take target, 0 = take source). Used for both NCHW latent blending (SDXL 4-channel, SD3 16-channel) and pixel-space recomposite (3-channel) — the math is identical, only the channel count differs.</summary>
    public static void BlendChannelsInPlace(Tensor target, Tensor source, Tensor mask)
    {
        if (target.Shape.Rank != 4 || source.Shape.Rank != 4 || mask.Shape.Rank != 4)
        {
            throw new ArgumentException("BlendChannelsInPlace requires [1, C, H, W] tensors.");
        }
        int channels = (int)target.Shape[1];
        int h = (int)target.Shape[2];
        int w = (int)target.Shape[3];
        if (source.Shape[1] != channels || source.Shape[2] != h || source.Shape[3] != w)
        {
            throw new ArgumentException($"Source shape {source.Shape} must match target shape {target.Shape}.");
        }
        if (mask.Shape[1] != 1 || mask.Shape[2] != h || mask.Shape[3] != w)
        {
            throw new ArgumentException($"Mask shape {mask.Shape} must be [1, 1, {h}, {w}] (matching target spatial dims).");
        }
        int spatial = h * w;
        float* tp = (float*)target.DataPointer;
        float* sp = (float*)source.DataPointer;
        float* mp = (float*)mask.DataPointer;
        for (int c = 0; c < channels; c++)
        {
            int chOff = c * spatial;
            for (int s = 0; s < spatial; s++)
            {
                float m = mp[s];
                tp[chOff + s] = tp[chOff + s] * m + sp[chOff + s] * (1f - m);
            }
        }
    }

    /// <summary>Packs an unpacked latent mask <c>[1, 1, latentH, latentW]</c> into Flux's 2×2-patchify packed form <c>[1, hPacked·wPacked, 4]</c>. Each packed cell holds the 4 sub-patch mask values in the same (top-left, top-right, bottom-left, bottom-right) order Flux's <c>PackLatent</c> uses for the latent itself, so a per-feature blend simply matches index <c>c*4 + p</c> against packed-mask index <c>p</c>.</summary>
    public static Tensor PackLatentMask2x2(Tensor latentMask, int latentH, int latentW)
    {
        if (latentMask.Shape.Rank != 4 || latentMask.Shape[0] != 1 || latentMask.Shape[1] != 1
            || latentMask.Shape[2] != latentH || latentMask.Shape[3] != latentW)
        {
            throw new ArgumentException(
                $"latentMask must be [1, 1, {latentH}, {latentW}]; got {latentMask.Shape}.", nameof(latentMask));
        }
        if ((latentH & 1) != 0 || (latentW & 1) != 0)
        {
            throw new ArgumentException($"Packed (2×2) mask requires even latent dims; got {latentH}x{latentW}.");
        }
        int hPacked = latentH / 2;
        int wPacked = latentW / 2;
        int seqLen = hPacked * wPacked;
        Tensor packed = new Tensor(new TensorShape(1, seqLen, 4), DType.F32);
        float* sp = (float*)latentMask.DataPointer;
        float* dp = (float*)packed.DataPointer;
        for (int ph = 0; ph < hPacked; ph++)
        {
            for (int pw = 0; pw < wPacked; pw++)
            {
                int outBase = (ph * wPacked + pw) * 4;
                dp[outBase + 0] = sp[(ph * 2 + 0) * latentW + (pw * 2 + 0)];
                dp[outBase + 1] = sp[(ph * 2 + 0) * latentW + (pw * 2 + 1)];
                dp[outBase + 2] = sp[(ph * 2 + 1) * latentW + (pw * 2 + 0)];
                dp[outBase + 3] = sp[(ph * 2 + 1) * latentW + (pw * 2 + 1)];
            }
        }
        return packed;
    }

    /// <summary>In-place mask blend for Flux's packed latent format <c>[1, seqLen, channels·4]</c> against a same-shape source: <c>target = target * mask + source * (1 - mask)</c>. The packed mask is <c>[1, seqLen, 4]</c> — one mask value per 2×2 sub-patch, broadcast across all latent channels via index <c>c*4 + p</c>.</summary>
    public static void BlendPackedInPlace(Tensor target, Tensor source, Tensor packedMask)
    {
        if (target.Shape.Rank != 3 || source.Shape.Rank != 3 || packedMask.Shape.Rank != 3)
        {
            throw new ArgumentException("BlendPackedInPlace requires [1, seqLen, F] tensors.");
        }
        if (target.Shape[0] != 1 || source.Shape[0] != 1 || packedMask.Shape[0] != 1)
        {
            throw new ArgumentException("BlendPackedInPlace requires batch size 1.");
        }
        long seqLen = target.Shape[1];
        long featDim = target.Shape[2];
        if (source.Shape[1] != seqLen || source.Shape[2] != featDim)
        {
            throw new ArgumentException($"Source shape {source.Shape} must match target shape {target.Shape}.");
        }
        if (packedMask.Shape[1] != seqLen || packedMask.Shape[2] != 4)
        {
            throw new ArgumentException($"Packed mask shape {packedMask.Shape} must be [1, {seqLen}, 4].");
        }
        if ((featDim & 3) != 0)
        {
            throw new ArgumentException($"Feature dim {featDim} must be a multiple of 4 for 2×2 patchified packing.");
        }
        long channels = featDim / 4;
        float* tp = (float*)target.DataPointer;
        float* sp = (float*)source.DataPointer;
        float* mp = (float*)packedMask.DataPointer;
        for (long s = 0; s < seqLen; s++)
        {
            long maskBase = s * 4;
            long featBase = s * featDim;
            for (long c = 0; c < channels; c++)
            {
                long chOff = featBase + c * 4;
                tp[chOff + 0] = tp[chOff + 0] * mp[maskBase + 0] + sp[chOff + 0] * (1f - mp[maskBase + 0]);
                tp[chOff + 1] = tp[chOff + 1] * mp[maskBase + 1] + sp[chOff + 1] * (1f - mp[maskBase + 1]);
                tp[chOff + 2] = tp[chOff + 2] * mp[maskBase + 2] + sp[chOff + 2] * (1f - mp[maskBase + 2]);
                tp[chOff + 3] = tp[chOff + 3] * mp[maskBase + 3] + sp[chOff + 3] * (1f - mp[maskBase + 3]);
            }
        }
    }
}
