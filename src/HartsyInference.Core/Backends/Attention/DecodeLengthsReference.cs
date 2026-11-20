using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>The definition of a decode over a cache whose sequences have their own lengths: each batch item attends the
/// first <c>lengths[b]</c> rows of its cache, the rows past that are never read, and every row's log-sum-exp can be
/// returned. Written from Comfy Kitchen's torch definition at the pinned commit (benchmarks/attention/protocol.json) and
/// checked against its outputs.</summary>
/// <remarks>
/// <para>It is the contract's attention (docs/ATTENTION_CONTRACT.md) run once per sequence, and is computed that way,
/// through <see cref="AttentionHostReference"/>: sequence <c>b</c> is one call over its own queries, its own cache with
/// <c>lengths[b]</c> valid keys, and its own position.</para>
/// <para>Causal: the step's <c>S</c> rows are the last <c>S</c> slots of the sequence, so row <c>j</c> sits at position
/// <c>lengths[b] − S + j</c> and attends the slots up to and including it. With one row that is plain decode. A row whose
/// position is below zero attends nothing.</para>
/// <para>Not causal: every row attends all <c>lengths[b]</c> slots, which is how a verify step reads its committed prefix
/// before <see cref="DecodeMergeReference"/> folds the step's own rows in.</para>
/// <para>A row that attends nothing is written as zeros and its LSE is negative infinity. Operands are F32, F16 or BF16;
/// the arithmetic is the host reference's, in double precision.</para>
/// </remarks>
public static class DecodeLengthsReference
{
    /// <summary>Runs the decode.</summary>
    /// <param name="output">Output, <c>[B, S, H × D]</c>.</param>
    /// <param name="query">Queries, <c>[B, H, S, D]</c>.</param>
    /// <param name="key">The cache's keys, <c>[B, Hkv, capacity, D]</c>.</param>
    /// <param name="value">The cache's values, <c>[B, Hkv, capacity, D]</c>.</param>
    /// <param name="lengths">Valid cache rows per sequence; a value below zero reads as zero and one above the capacity
    /// as the capacity.</param>
    /// <param name="causal">Whether the step's rows are the sequence's last rows and each attends only up to itself.</param>
    /// <param name="lse">When not empty, receives the natural-log LSE of every row's scaled scores as <c>[B, H, S]</c>.</param>
    /// <param name="scale">The score multiplier; null is <c>D^-1/2</c>.</param>
    /// <exception cref="ArgumentException">An operand does not fit the call.</exception>
    public static void Run(Tensor output, Tensor query, Tensor key, Tensor value, ReadOnlySpan<int> lengths, bool causal,
        Span<float> lse = default, float? scale = null)
    {
        if (query.Shape.Rank != 4 || key.Shape.Rank != 4 || key.Shape != value.Shape)
            throw new ArgumentException($"A decode takes Q [B, H, S, D] and K, V [B, Hkv, capacity, D]; got Q={query.Shape}, K={key.Shape}, V={value.Shape}.");
        int batch = (int)query.Shape[0], heads = (int)query.Shape[1], rows = (int)query.Shape[2], dim = (int)query.Shape[3];
        int kvHeads = (int)key.Shape[1], capacity = (int)key.Shape[2];
        if (query.ElementCount == 0 || key.ElementCount == 0 || key.Shape[0] != batch || key.Shape[3] != dim || heads % kvHeads != 0)
            throw new ArgumentException($"The cache does not match the queries; got Q={query.Shape}, K={key.Shape}.");
        TensorShape expected = new(batch, rows, heads * dim);
        if (output.Shape != expected) throw new ArgumentException($"The output is [B, S, H × D] = {expected}; got {output.Shape}.", nameof(output));
        if (lengths.Length != batch) throw new ArgumentException($"There are {lengths.Length} lengths for {batch} sequences.", nameof(lengths));
        if (!lse.IsEmpty && lse.Length != batch * heads * rows)
            throw new ArgumentException($"LSE holds {lse.Length} values; [B, H, S] needs {batch * heads * rows}.", nameof(lse));
        float alpha = scale ?? 1f / MathF.Sqrt(dim);
        if (!float.IsFinite(alpha)) throw new ArgumentException("The scale must be finite.", nameof(scale));

        long rowWidth = (long)heads * dim;
        float[] sequenceLse = lse.IsEmpty ? [] : new float[heads * rows];
        for (int b = 0; b < batch; b++)
        {
            int length = Math.Clamp(lengths[b], 0, capacity);
            // The first row that attends anything: under the causal rule row j sits at position length − S + j.
            int first = causal ? Math.Clamp(rows - length, 0, rows) : length == 0 ? rows : 0;
            if (first > 0)
            {
                AttentionOperand empty = new(output, AttentionLayout.TokenMajor, 1, heads, first, dim, b * rows * rowWidth, rows * rowWidth, dim, rowWidth);
                for (int h = 0; h < heads; h++)
                    for (int r = 0; r < first; r++)
                    {
                        for (int c = 0; c < dim; c++) AttentionHostReference.Write(in empty, 0, h, r, c, 0f);
                        if (!lse.IsEmpty) lse[(b * heads + h) * rows + r] = float.NegativeInfinity;
                    }
            }
            int live = rows - first;
            if (live == 0) continue;
            AttentionCall call = new()
            {
                Query = new(query, AttentionLayout.HeadMajor, 1, heads, live, dim, ((long)b * heads * rows + first) * dim, (long)heads * rows * dim,
                    (long)rows * dim, dim),
                Key = Cache(key, b, kvHeads, capacity, dim),
                Value = Cache(value, b, kvHeads, capacity, dim),
                Output = new(output, AttentionLayout.TokenMajor, 1, heads, live, dim, ((long)b * rows + first) * rowWidth, rows * rowWidth, dim, rowWidth),
                Scale = alpha,
                Mask = AttentionMask.FromFlash(causal, 0),
                Positions = AttentionPositions.Host(length, causal ? length - rows + first : 0),
            };
            Span<float> callLse = lse.IsEmpty ? default : sequenceLse.AsSpan(0, heads * live);
            AttentionHostReference.Run(in call, callLse);
            if (lse.IsEmpty) continue;
            for (int h = 0; h < heads; h++)
                callLse.Slice(h * live, live).CopyTo(lse.Slice((b * heads + h) * rows + first, live));
        }
    }

    private static AttentionOperand Cache(Tensor cache, int b, int kvHeads, int capacity, int dim) =>
        new(cache, AttentionLayout.HeadMajor, 1, kvHeads, capacity, dim, (long)b * kvHeads * capacity * dim, (long)kvHeads * capacity * dim,
            (long)capacity * dim, dim);
}
