using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>CPU reference for <see cref="IBackend.TopKLastDim"/>; CUDA is tested against it.</summary>
public static class TopKReference
{
    /// <summary>Checks tensors and k for TopKLastDim; shared by every backend.</summary>
    public static void Validate(Tensor values, Tensor indices, Tensor input, int k, Tensor? validLengths)
    {
        if (input.Shape.Rank < 1) throw new ArgumentException("TopKLastDim input must be at least rank 1.", nameof(input));
        long n = input.Shape[input.Shape.Rank - 1];
        if (k < 1 || k > n) throw new ArgumentOutOfRangeException(nameof(k), $"TopKLastDim k must be in [1,{n}]; got {k}.");
        if (input.DType != DType.F32 || values.DType != DType.F32)
            throw new NotSupportedException("TopKLastDim supports F32 input and values only.");
        if (indices.DType != DType.I32) throw new NotSupportedException("TopKLastDim requires I32 indices.");
        long rows = input.ElementCount / n;
        if (values.ElementCount != rows * k || indices.ElementCount != rows * k)
            throw new ArgumentException($"TopKLastDim outputs must hold {rows}x{k} entries.");
        if (validLengths is not null && (validLengths.DType != DType.I32 || validLengths.ElementCount != rows))
            throw new ArgumentException($"TopKLastDim validLengths must be I32 with {rows} entries.", nameof(validLengths));
    }

    /// <summary>Per row: the k largest of the first validLengths[row] entries (all when null), ties resolved to the lowest index.</summary>
    /// <remarks>Output is value-descending, or index-ascending when <paramref name="sortByIndex"/>. Slots beyond the valid
    /// length hold index -1 and value -inf. NaN inputs rank below every number.</remarks>
    public static unsafe void Apply(Tensor values, Tensor indices, Tensor input, int k, Tensor? validLengths, bool sortByIndex)
    {
        Validate(values, indices, input, k, validLengths);
        int n = (int)input.Shape[input.Shape.Rank - 1];
        long rows = input.ElementCount / n;
        float* x = (float*)input.DataPointer, vo = (float*)values.DataPointer;
        int* io = (int*)indices.DataPointer;
        int* valid = validLengths is null ? null : (int*)validLengths.DataPointer;
        int[] order = new int[n];
        for (long r = 0; r < rows; r++)
        {
            int len = valid == null ? n : Math.Clamp(valid[r], 0, n);
            float* row = x + r * n;
            for (int i = 0; i < len; i++) order[i] = i;
            Array.Sort(order, 0, len, Comparer<int>.Create((a, b) => Compare(row, a, b)));
            int take = Math.Min(k, len);
            if (sortByIndex) Array.Sort(order, 0, take);
            for (int j = 0; j < k; j++)
            {
                bool live = j < take;
                io[r * k + j] = live ? order[j] : -1;
                vo[r * k + j] = live ? row[order[j]] : float.NegativeInfinity;
            }
        }
    }

    private static unsafe int Compare(float* row, int a, int b)
    {
        float va = row[a], vb = row[b];
        bool na = float.IsNaN(va), nb = float.IsNaN(vb);
        if (na != nb) return na ? 1 : -1;
        if (!na && va != vb) return va > vb ? -1 : 1;
        return a.CompareTo(b);
    }
}
