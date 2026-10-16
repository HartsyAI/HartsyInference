using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Loads RMSNorm weights for the standard norm op, baking in the zero-centered <c>(1 + w)</c> scale Gemma, Qwen3.5 and
/// Fish Audio's Drama 3 store, so the runtime norm stays the plain one and no borrowed mmap or cached weight is mutated.</summary>
public static unsafe class NormWeights
{
    /// <summary>A new F32 tensor holding <c>1 + weight</c>, which the caller owns.</summary>
    public static Tensor OnePlus(Tensor weight)
    {
        ArgumentNullException.ThrowIfNull(weight);
        Tensor f = ProjectionOps.EnsureF32(weight);
        try
        {
            Tensor output = new(f.Shape, DType.F32);
            long n = f.ElementCount;
            float* src = (float*)f.DataPointer;
            float* dst = (float*)output.DataPointer;
            for (long i = 0; i < n; i++) dst[i] = src[i] + 1f;
            return output;
        }
        finally
        {
            if (!ReferenceEquals(f, weight)) f.Dispose();
        }
    }

    /// <summary>The weight as F32: <see cref="OnePlus"/> when <paramref name="addOne"/>, else the weight itself when already F32 or a new
    /// F32 copy (callers check <c>ReferenceEquals</c> before disposing).</summary>
    public static Tensor Load(Tensor weight, bool addOne) => addOne ? OnePlus(weight) : ProjectionOps.EnsureF32(weight);
}
