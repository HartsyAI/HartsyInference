using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Device-footprint sums shared by the pipeline VRAM planners.</summary>
internal static class WeightBytes
{
    /// <summary>Sums device bytes via <see cref="DType.ComputeByteCount"/> — block-quantized dtypes report <c>SizeInBytes == 0</c>, so the naive <c>ElementCount * SizeInBytes</c> product would total to zero and silently disable streaming.</summary>
    internal static long Sum(IEnumerable<Tensor> tensors)
    {
        long total = 0;
        foreach (Tensor t in tensors) total += t.DType.ComputeByteCount(t.ElementCount);
        return total;
    }
}
