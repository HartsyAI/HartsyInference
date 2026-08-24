using System.Collections.Generic;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>Builds the extended text-conditioning stream for regional generation in joint-attention DiTs (Flux, Z-Image): each region's encoded conditioning is appended after the base conditioning along the sequence axis, and the per-region column ranges + image-grid masks needed by <see cref="RegionalAttentionBias"/> are returned. The image tokens live in a separate stream, so column ranges are relative to the conditioning stream (offset 0).</summary>
public static class RegionalConditioningLayout
{
    /// <summary>Concatenates <paramref name="baseCond"/> <c>[1, baseLen, D]</c> with each region's <c>[1, rLen, D]</c> conditioning. Returns the extended tensor, its sequence length, each region's column range within it, and each region's image-grid mask (row-major <c>[gridH*gridW]</c>).</summary>
    public static (Tensor Extended, int ExtLen, List<(int Start, int End)> Ranges, List<float[]> GridMasks)
        BuildTextStream(RegionalPlan plan, Tensor baseCond, int gridH, int gridW)
    {
        int baseLen = (int)baseCond.Shape[1];
        int dim = (int)baseCond.Shape[2];
        int numImg = gridH * gridW;
        int extLen = baseLen;
        foreach (RegionConditioning region in plan.Regions)
        {
            if ((int)region.Cond.Shape[2] != dim)
            {
                throw new ArgumentException($"Region conditioning dim {region.Cond.Shape[2]} != base dim {dim}.");
            }
            extLen += (int)region.Cond.Shape[1];
        }
        Tensor extended = new Tensor(new TensorShape(1, extLen, dim), DType.F32);
        List<(int Start, int End)> ranges = new List<(int Start, int End)>(plan.Regions.Count);
        List<float[]> masks = new List<float[]>(plan.Regions.Count);
        CopyStream(baseCond, extended, baseLen, dim, plan, gridH, gridW, numImg, ranges, masks);
        return (extended, extLen, ranges, masks);
    }

    private static unsafe void CopyStream(Tensor baseCond, Tensor extended, int baseLen, int dim, RegionalPlan plan,
        int gridH, int gridW, int numImg, List<(int Start, int End)> ranges, List<float[]> masks)
    {
        float* dst = (float*)extended.DataPointer;
        float* src = (float*)baseCond.DataPointer;
        long baseBytes = (long)baseLen * dim * sizeof(float);
        Buffer.MemoryCopy(src, dst, baseBytes, baseBytes);
        long cursor = baseLen;
        foreach (RegionConditioning region in plan.Regions)
        {
            int len = (int)region.Cond.Shape[1];
            float* regionSrc = (float*)region.Cond.DataPointer;
            long regionBytes = (long)len * dim * sizeof(float);
            Buffer.MemoryCopy(regionSrc, dst + cursor * dim, regionBytes, regionBytes);
            ranges.Add(((int)cursor, (int)cursor + len));
            cursor += len;
            Tensor latentMask = region.Mask.ToLatentMask(gridH, gridW);
            float[] grid = latentMask.AsReadOnlySpan<float>().ToArray();
            latentMask.Dispose();
            if (grid.Length != numImg)
            {
                throw new ArgumentException($"Region grid mask length {grid.Length} != numImg {numImg}.");
            }
            masks.Add(grid);
        }
    }
}
