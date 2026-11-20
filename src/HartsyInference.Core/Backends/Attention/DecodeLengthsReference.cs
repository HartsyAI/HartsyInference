using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>The definition of a decode over a cache whose sequences have their own lengths: each batch item attends the
/// first <c>lengths[b]</c> rows of its cache, the rows past that are never read, and every row's log-sum-exp can be
/// returned. Written from Comfy Kitchen's torch definition at the pinned commit (benchmarks/attention/protocol.json) and
/// checked against its outputs.</summary>
/// <remarks>
/// <para>It is the contract's attention with a length per sequence (<see cref="AttentionCall.SequenceLengths"/>,
/// docs/ATTENTION_CONTRACT.md §2), in the shapes a decode has, and <see cref="AttentionHostReference"/> computes it:
/// sequence <c>b</c> is one call over its own queries, its own cache with <c>lengths[b]</c> valid keys, and its own
/// position.</para>
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
    /// <exception cref="AttentionUnsupportedException">An operand is stored in a type the host reference does not read.</exception>
    /// <remarks>The whole call is checked before anything is written, whatever the lengths are.</remarks>
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

        AttentionCall call = AttentionCall.FromDecodeWithLengths(output, query, key, value, lengths.ToArray(), causal, alpha);
        AttentionHostReference.Run(in call, lse);
    }
}
