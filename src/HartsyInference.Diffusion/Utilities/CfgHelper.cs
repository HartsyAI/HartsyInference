using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Shared helpers for classifier-free guidance pipelines that run a UNet/transformer twice per step (uncond + cond) and combine the two outputs. Centralizes the batch-slice + CFG-combine operations duplicated across SD1.5, SDXL, SD3, and every legacy CFG pipeline.</summary>
public static unsafe class CfgHelper
{
    /// <summary>Extracts a single element along the batch dimension of a 3-D tensor [B, seqLen, hiddenSize] → [1, seqLen, hiddenSize]. F32 only — callers needing other dtypes should cast first.</summary>
    public static Tensor SliceBatchElement(Tensor tensor, int batchIdx, int seqLen, int hiddenSize)
    {
        TensorShape shape = new TensorShape(1, seqLen, hiddenSize);
        Tensor slice = new Tensor(shape, DType.F32);
        float* srcPtr = (float*)tensor.DataPointer;
        float* dstPtr = (float*)slice.DataPointer;
        int elements = seqLen * hiddenSize;
        int srcOffset = batchIdx * elements;
        for (int i = 0; i < elements; i++)
        {
            dstPtr[i] = srcPtr[srcOffset + i];
        }
        return slice;
    }

    /// <summary>Extracts a single element along the batch dimension of a 2-D tensor [B, dim] → [1, dim]. F32 only.</summary>
    public static Tensor SliceBatchElement1D(Tensor tensor, int batchIdx, int dim)
    {
        TensorShape shape = new TensorShape(1, dim);
        Tensor slice = new Tensor(shape, DType.F32);
        float* srcPtr = (float*)tensor.DataPointer;
        float* dstPtr = (float*)slice.DataPointer;
        int srcOffset = batchIdx * dim;
        for (int i = 0; i < dim; i++)
        {
            dstPtr[i] = srcPtr[srcOffset + i];
        }
        return slice;
    }

    /// <summary>Whether classifier-free guidance has any effect at this scale. At <c>scale ≤ 1</c> the combine reduces to the conditional output (<c>uncond + 1·(cond − uncond) = cond</c>), so the unconditional forward pass is wasted work. Guidance-distilled checkpoints (Flux-dev, SDXL-Turbo, LCM/TCD) run at scale 0-1; gating the uncond pass on this halves per-step compute and activation memory for them. A small epsilon avoids running a pass that contributes nothing at scale ≈ 1.</summary>
    public static bool IsGuidanceActive(float scale) => scale > 1.0f + 1e-4f;

    /// <summary>Applies classifier-free guidance: <c>output = uncond + scale * (cond - uncond)</c> (the standard convention, anchored on the unconditional output). Both inputs must be F32 and have identical shape; output is allocated F32 with that same shape. The inputs are NOT disposed — caller owns the lifetime.</summary>
    public static Tensor ApplyCfg(Tensor uncond, Tensor cond, float scale) => CombineCfg(uncond, cond, scale, condAnchored: false);

    /// <summary>Applies cond-anchored classifier-free guidance: <c>output = cond + scale * (cond - uncond)</c>. This is algebraically <see cref="ApplyCfg"/> with the anchor swapped to the conditional output, the convention used by Krea 2 and Chroma. Because the baseline is <c>cond</c> rather than <c>uncond</c>, a given guidance_scale produces stronger guidance than the standard convention: e.g. Krea 2's <c>guidance_scale = 4.5</c> here ≈ <c>5.5</c> under <see cref="ApplyCfg"/> (the cond-anchored formula adds one extra unit of <c>(cond - uncond)</c> relative to the uncond-anchored one). Both inputs must be F32 and have identical shape; output is allocated F32 with that same shape. The inputs are NOT disposed — caller owns the lifetime.</summary>
    // VALIDATION-PENDING: verify cond-anchored formula vs Krea 2 / Chroma reference pipelines (cond + scale*(cond - uncond)).
    public static Tensor ApplyCfgCondAnchored(Tensor cond, Tensor uncond, float scale) => CombineCfg(uncond, cond, scale, condAnchored: true);

    /// <summary>Core CFG combine shared by <see cref="ApplyCfg"/> and <see cref="ApplyCfgCondAnchored"/>. When <paramref name="condAnchored"/> is false: <c>uncond + scale*(cond - uncond)</c>; when true: <c>cond + scale*(cond - uncond)</c>.</summary>
    private static Tensor CombineCfg(Tensor uncond, Tensor cond, float scale, bool condAnchored)
    {
        if (uncond.DType != DType.F32 || cond.DType != DType.F32)
            throw new ArgumentException($"ApplyCfg requires F32 inputs; got uncond={uncond.DType}, cond={cond.DType}. Cast via DtypeCastHelper.EnsureF32 first.");
        if (!uncond.Shape.Equals(cond.Shape))
            throw new ArgumentException($"ApplyCfg shape mismatch: uncond={uncond.Shape}, cond={cond.Shape}.");
        Tensor output = new Tensor(uncond.Shape, DType.F32);
        float* uncPtr = (float*)uncond.DataPointer;
        float* conPtr = (float*)cond.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)uncond.ElementCount;
        for (int i = 0; i < count; i++)
        {
            float anchor = condAnchored ? conPtr[i] : uncPtr[i];
            outPtr[i] = anchor + scale * (conPtr[i] - uncPtr[i]);
        }
        return output;
    }

    /// <summary>Applies classifier-free guidance with Lumina-2.0's <c>cfg_normalization</c>: combine <c>guided = uncond + scale·(cond − uncond)</c>, then rescale the guided prediction back to the conditional's per-token L2 norm over the last dimension — <c>guided ·= ‖cond‖₂ / (‖guided‖₂ + eps)</c>, where each norm reduces over the final axis and is broadcast across it.
    /// <para>VALIDATION-PENDING: verify against diffusers <c>Lumina2Pipeline</c> (<c>pipeline_lumina2.py</c> lines 750-755): <c>cond_norm = torch.norm(noise_pred_cond, dim=-1, keepdim=True)</c>; <c>noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)</c>; <c>noise_pred = noise_pred * (cond_norm / noise_norm)</c>. Diffusers uses no epsilon; the <paramref name="eps"/> here guards the division and is negligible at the default 1e-12. The reduction axis is the last tensor dim — the W axis for the unpatchified <c>[B, C, H, W]</c> velocity.</para>
    /// Both inputs must be F32 with identical shape; output is a new F32 tensor with that shape. Inputs are NOT disposed — caller owns the lifetime.</summary>
    public static Tensor ApplyCfgNormalized(Tensor uncond, Tensor cond, float scale, float eps = 1e-12f)
    {
        if (uncond.DType != DType.F32 || cond.DType != DType.F32)
            throw new ArgumentException($"ApplyCfgNormalized requires F32 inputs; got uncond={uncond.DType}, cond={cond.DType}. Cast via DtypeCastHelper.EnsureF32 first.");
        if (!uncond.Shape.Equals(cond.Shape))
            throw new ArgumentException($"ApplyCfgNormalized shape mismatch: uncond={uncond.Shape}, cond={cond.Shape}.");

        Tensor output = new Tensor(uncond.Shape, DType.F32);
        float* uncPtr = (float*)uncond.DataPointer;
        float* conPtr = (float*)cond.DataPointer;
        float* outPtr = (float*)output.DataPointer;

        // The last dimension is the per-token feature axis the L2 norm reduces over; everything before it is the token grid.
        int lastDim = (int)uncond.Shape[uncond.Shape.Rank - 1];
        long tokens = uncond.ElementCount / lastDim;

        for (long tok = 0; tok < tokens; tok++)
        {
            long baseOffset = tok * lastDim;

            // Pass 1: combine guidance and accumulate ‖cond‖₂ and ‖guided‖₂ over this token's last-dim slice.
            double condSq = 0.0;
            double guidedSq = 0.0;
            for (int d = 0; d < lastDim; d++)
            {
                long idx = baseOffset + d;
                float unc = uncPtr[idx];
                float con = conPtr[idx];
                float guided = unc + scale * (con - unc);
                outPtr[idx] = guided;
                condSq += (double)con * con;
                guidedSq += (double)guided * guided;
            }

            // Pass 2: rescale guided back to the conditional's L2 norm, broadcast across the last dim.
            float ratio = (float)(Math.Sqrt(condSq) / (Math.Sqrt(guidedSq) + eps));
            for (int d = 0; d < lastDim; d++)
                outPtr[baseOffset + d] *= ratio;
        }
        return output;
    }

    /// <summary>Concatenates two [B, seqLen, dimA] and [B, seqLen, dimB] tensors along the last dimension → [B, seqLen, dimA + dimB]. Used by SDXL-family pipelines to stitch CLIP-L and CLIP-G hidden states into the 2048-wide text embedding the UNet expects. F32 only.</summary>
    public static Tensor ConcatLastDim(Tensor a, Tensor b)
    {
        if (a.Shape.Rank != 3 || b.Shape.Rank != 3)
            throw new ArgumentException($"ConcatLastDim expects rank-3 tensors; got {a.Shape.Rank} and {b.Shape.Rank}.");
        if (a.Shape[0] != b.Shape[0] || a.Shape[1] != b.Shape[1])
            throw new ArgumentException($"ConcatLastDim requires matching [batch, seqLen]; got {a.Shape} vs {b.Shape}.");
        int batch = (int)a.Shape[0];
        int seqLen = (int)a.Shape[1];
        int dimA = (int)a.Shape[2];
        int dimB = (int)b.Shape[2];
        int dimOut = dimA + dimB;
        Tensor output = new Tensor(new TensorShape(batch, seqLen, dimOut), DType.F32);
        float* aPtr = (float*)a.DataPointer;
        float* bPtr = (float*)b.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        for (int bIdx = 0; bIdx < batch; bIdx++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int aOffset = (bIdx * seqLen + s) * dimA;
                int bOffset = (bIdx * seqLen + s) * dimB;
                int outOffset = (bIdx * seqLen + s) * dimOut;
                for (int d = 0; d < dimA; d++) outPtr[outOffset + d] = aPtr[aOffset + d];
                for (int d = 0; d < dimB; d++) outPtr[outOffset + dimA + d] = bPtr[bOffset + d];
            }
        }
        return output;
    }
}
