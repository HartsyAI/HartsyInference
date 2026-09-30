using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>CPU reference for the sliding-window ring slots each query attends to (<c>get_window_topk_idxs</c>).</summary>
public static class WindowIndicesReference
{
    /// <summary>Row and column counts of the index matrix: one row per query when prefilling, one row when decoding.</summary>
    public static (int Rows, int Cols) Shape(int windowSize, int seqLen, int startPos) =>
        startPos == 0 ? (seqLen, Math.Min(seqLen, windowSize)) : (1, windowSize);

    /// <summary>Checks the operands of <see cref="Apply"/>; shared by every backend.</summary>
    public static void Validate(Tensor indices, int windowSize, int seqLen, int startPos)
    {
        if (windowSize < 1) throw new ArgumentOutOfRangeException(nameof(windowSize), "windowSize must be positive.");
        if (seqLen < 1) throw new ArgumentOutOfRangeException(nameof(seqLen), "seqLen must be positive.");
        if (startPos < 0) throw new ArgumentOutOfRangeException(nameof(startPos), "startPos must not be negative.");
        if (startPos > 0 && seqLen != 1) throw new ArgumentException("A decode step (startPos > 0) takes exactly one query.", nameof(seqLen));
        if (indices.DType != DType.I32) throw new ArgumentException("indices must be I32.", nameof(indices));
        (int rows, int cols) = Shape(windowSize, seqLen, startPos);
        if (indices.ElementCount != (long)rows * cols)
            throw new ArgumentException($"indices must hold {rows}x{cols} entries; got {indices.ElementCount}.", nameof(indices));
    }

    /// <summary>Fills <c>[rows, cols]</c> ring slots, -1 where a slot holds nothing yet.</summary>
    /// <remarks>Prefill (<paramref name="startPos"/> 0): query <c>q</c> sees positions <c>max(q-W+1,0)..q</c>, so slots are
    /// positions. Decode: the whole ring listed oldest first, with slots beyond <paramref name="startPos"/> empty.</remarks>
    public static unsafe void Apply(Tensor indices, int windowSize, int seqLen, int startPos)
    {
        Validate(indices, windowSize, seqLen, startPos);
        (int rows, int cols) = Shape(windowSize, seqLen, startPos);
        int* p = (int*)indices.DataPointer;
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                p[(long)r * cols + c] = Slot(windowSize, startPos, r, c);
    }

    /// <summary>The slot at row <paramref name="row"/>, column <paramref name="col"/>; the CUDA kernel mirrors this.</summary>
    public static int Slot(int windowSize, int startPos, int row, int col)
    {
        if (startPos == 0)
        {
            int idx = Math.Max(row - windowSize + 1, 0) + col;
            return idx > row ? -1 : idx;
        }
        int oldest = startPos % windowSize + 1;
        int slot = col < windowSize - oldest ? oldest + col : col - (windowSize - oldest);
        return slot > startPos ? -1 : slot;
    }
}
