using HartsyInference.Core.Numerics;
using System.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Duo;

/// <summary>Merges one Duo LoRA module into rows of a backbone weight the way upstream's <c>load_effective</c> does: <c>delta = (B @ A) * scale</c> in F32, rounded to the weight's dtype, then added to the weight in that dtype
/// (<c>local.add_((b @ a * scale).to(local.dtype))</c>). For a BF16 weight that is two roundings, delta then sum, both to nearest even; an F32 weight takes the delta as is.
/// The F32 product sums over the rank in a different order than torch's GEMM, so a delta can differ from upstream's in the last F32 bit, which rounds to a different BF16 value only where the product sits next to a rounding boundary.</summary>
public static class DuoLoraMerge
{
    /// <summary>Adds the module into <paramref name="weight"/>'s rows starting at <paramref name="rowStart"/>, in place. <paramref name="loraA"/> is <c>[rank, in]</c> and <paramref name="loraB"/> is <c>[rows, rank]</c>, both F32;
    /// <paramref name="swapHalves"/> reads B's rows from the other half first, for a B in <c>[value; gate]</c> order against a <c>[gate; value]</c> weight.</summary>
    public static unsafe void MergeRows(Tensor weight, long rowStart, Tensor loraA, Tensor loraB, float scale, bool swapHalves)
    {
        ArgumentNullException.ThrowIfNull(weight);
        ArgumentNullException.ThrowIfNull(loraA);
        ArgumentNullException.ThrowIfNull(loraB);
        if (weight.DType != DType.BF16 && weight.DType != DType.F32)
            throw new NotSupportedException($"A LoRA merges into a BF16 or F32 weight, not {weight.DType}.");
        if (loraA.DType != DType.F32 || loraB.DType != DType.F32)
            throw new ArgumentException($"LoRA A and B must be F32, got {loraA.DType} and {loraB.DType}.");
        if (weight.Shape.Rank != 2 || loraA.Shape.Rank != 2 || loraB.Shape.Rank != 2)
            throw new ArgumentException("Weight and LoRA matrices must be rank 2.");

        long rows = loraB.Shape[0], rank = loraB.Shape[1], inDim = weight.Shape[1];
        if (loraA.Shape[0] != rank || loraA.Shape[1] != inDim)
            throw new ArgumentException($"LoRA A {loraA.Shape} does not match B {loraB.Shape} and weight {weight.Shape}.");
        if (rowStart < 0 || rowStart + rows > weight.Shape[0])
            throw new ArgumentException($"LoRA rows [{rowStart}, {rowStart + rows}) do not fit a weight of {weight.Shape[0]} rows.");
        if (swapHalves && rows % 2 != 0)
            throw new ArgumentException("A swapped gate/value LoRA needs an even row count.");

        bool bf16 = weight.DType == DType.BF16;
        byte* basePtr = (byte*)weight.DataPointer;
        float* a = (float*)loraA.DataPointer;
        float* b = (float*)loraB.DataPointer;
        nint aAddress = (nint)a, bAddress = (nint)b, weightAddress = (nint)basePtr;
        long half = rows / 2;

        Parallel.For(0, rows, () => new float[inDim], (r, _, acc) =>
        {
            float* aPtr = (float*)aAddress;
            float* bPtr = (float*)bAddress;
            long sourceRow = swapHalves ? (r + half) % rows : r;
            Array.Clear(acc);
            for (long k = 0; k < rank; k++)
            {
                float coefficient = bPtr[sourceRow * rank + k];
                AddScaled(acc, aPtr + k * inDim, coefficient);
            }

            long target = (rowStart + r) * inDim;
            if (bf16)
            {
                ushort* row = (ushort*)weightAddress + target;
                for (long j = 0; j < inDim; j++)
                {
                    float delta = BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(acc[j] * scale));
                    row[j] = BFloat16Bits.FromFloat(BFloat16Bits.ToFloat(row[j]) + delta);
                }
            }
            else
            {
                float* row = (float*)weightAddress + target;
                for (long j = 0; j < inDim; j++) row[j] += acc[j] * scale;
            }
            return acc;
        }, _ => { });
    }


    private static unsafe void AddScaled(float[] acc, float* source, float coefficient)
    {
        int width = Vector<float>.Count, j = 0;
        fixed (float* dst = acc)
        {
            for (; j <= acc.Length - width; j += width)
            {
                Vector<float> sum = Vector.Load(dst + j) + Vector.Load(source + j) * coefficient;
                sum.Store(dst + j);
            }
            for (; j < acc.Length; j++) dst[j] += source[j] * coefficient;
        }
    }
}
