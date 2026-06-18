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

    /// <summary>Applies classifier-free guidance: <c>output = uncond + scale * (cond - uncond)</c>. Both inputs must be F32 and have identical shape; output is allocated F32 with that same shape. The inputs are NOT disposed — caller owns the lifetime.</summary>
    public static Tensor ApplyCfg(Tensor uncond, Tensor cond, float scale)
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
            outPtr[i] = uncPtr[i] + scale * (conPtr[i] - uncPtr[i]);
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
