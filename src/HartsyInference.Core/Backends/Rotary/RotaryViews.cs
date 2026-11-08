using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Rotary;

/// <summary>Builds the views of a <see cref="RotarySpec"/>. An activation base is a six-axis view
/// <c>[B, H, S1, S2, S3, D]</c> over the vectors to rotate; a table base is a six-axis view <c>[B, H, S1, S2, S3, E]</c>
/// over the entries, broadcast (stride 0) along every axis the table does not vary with. <see cref="Create"/> then
/// cuts the rotated width out of <c>D</c>, pairs its elements and maps each element onto its table entry.</summary>
public static class RotaryViews
{
    /// <summary>Token-major vectors <c>[B, L, H, D]</c>.</summary>
    public static StridedView TokenMajor(TensorShape shape)
    {
        RequireRank(shape, 4, "token-major [B, L, H, D]");
        return StridedView.Of(shape).Permute([0, 2, 1, 3]).Unsqueeze(3).Unsqueeze(3);
    }

    /// <summary>Head-major vectors <c>[B, H, L, D]</c>.</summary>
    public static StridedView HeadMajor(TensorShape shape)
    {
        RequireRank(shape, 4, "head-major [B, H, L, D]");
        return StridedView.Of(shape).Unsqueeze(3).Unsqueeze(3);
    }

    /// <summary>Rows <c>[S, heads·D]</c> (or any shape with that element count): one batch, heads packed in each row.</summary>
    public static StridedView Rows(long tokens, int heads, int headDim)
        => StridedView.Of([tokens, heads, headDim]).Unsqueeze(0).Permute([0, 2, 1, 3]).Unsqueeze(3).Unsqueeze(3);

    /// <summary>A grid of vectors <c>[1, T, H, W, heads, D]</c>, each sequence axis its own coordinate.</summary>
    public static StridedView Grid(TensorShape shape)
    {
        RequireRank(shape, 6, "grid [1, T, H, W, heads, D]");
        return StridedView.Of(shape).Permute([0, 4, 1, 2, 3, 5]);
    }

    /// <summary>A table <c>[L, E]</c> or <c>[B, L, E]</c> shared by every head: <c>[B, H, L, 1, 1, E]</c> with the head axis (and the batch axis of a two-axis table) broadcast.</summary>
    public static StridedView SharedTable(TensorShape shape, long batch, long heads)
    {
        StridedView rows = shape.Rank == 2 ? StridedView.Of(shape).Broadcast(0, batch)
            : shape.Rank == 3 ? StridedView.Of(shape)
            : throw new ArgumentException($"A shared table is [L, E] or [B, L, E]; got {shape}.", nameof(shape));
        if (rows[0] != batch)
            throw new ArgumentException($"The table {shape} does not cover batch {batch}.", nameof(shape));
        return rows.Broadcast(1, heads).Unsqueeze(3).Unsqueeze(3);
    }

    /// <summary>A table per head, <c>[H, L, E]</c>: <c>[B, H, L, 1, 1, E]</c> with the batch axis broadcast.</summary>
    public static StridedView PerHeadTable(TensorShape shape, long batch)
    {
        RequireRank(shape, 3, "per-head table [H, L, E]");
        return StridedView.Of(shape).Broadcast(0, batch).Unsqueeze(3).Unsqueeze(3);
    }

    /// <summary>A table whose rows pack one lane group per head, <c>[L, heads·lanes]</c> (LTX-2's split tables): <c>[1, H, L, 1, 1, lanes]</c>.</summary>
    public static StridedView PerHeadLanes(long tokens, int heads, int lanes)
        => StridedView.Of([tokens, heads, lanes]).Unsqueeze(0).Permute([0, 2, 1, 3]).Unsqueeze(3).Unsqueeze(3);

    /// <summary>A table indexed by one sequence axis of a grid, <c>[axisLength, E]</c>: broadcast over batch, heads and the other two axes.</summary>
    public static StridedView AxisTable(long batch, long heads, long s1, long s2, long s3, int seqAxis, long entries)
    {
        if ((uint)seqAxis >= 3)
            throw new ArgumentOutOfRangeException(nameof(seqAxis), seqAxis, "The sequence axis is 0, 1 or 2.");
        Span<long> strides = stackalloc long[6];
        strides[5] = 1;
        strides[2 + seqAxis] = entries;
        return new StridedView([batch, heads, s1, s2, s3, entries], strides);
    }

    /// <summary>The LTX-2 split rotation over rows <c>[S, heads·D]</c>: rotate-half within each head, one angle per pair, the
    /// angles packed per head in <c>[S, heads·D/2]</c> rows. The geometry the fused <see cref="IBackend.RmsNormRotary"/> kernels run.</summary>
    public static RotarySpec SplitRows(long tokens, int heads, int headDim)
        => Create(Rows(tokens, heads, headDim), PerHeadLanes(tokens, heads, headDim / 2), RotaryPairing.RotateHalf, RopeTableLayout.InterleavedPrefix);

    /// <summary>A six-axis activation or table base from explicit extents and strides.</summary>
    public static StridedView Explicit(ReadOnlySpan<long> dims, ReadOnlySpan<long> strides, long offset = 0)
    {
        if (dims.Length != 6)
            throw new ArgumentException($"A rotary base view has six axes [B, H, S1, S2, S3, D]; got {dims.Length}.", nameof(dims));
        return new StridedView(dims, strides, offset);
    }

    /// <summary>Pairs the rotated elements and maps each onto its table entries.</summary>
    /// <param name="vectors">Activation base <c>[B, H, S1, S2, S3, D]</c>.</param>
    /// <param name="table">Table base <c>[B, H, S1, S2, S3, E]</c>; the same geometry serves cos and sin.</param>
    /// <param name="pairing">Which elements pair up.</param>
    /// <param name="layout">How a table row stores its angles: <see cref="RopeTableLayout.RotateHalf"/> holds an entry per
    /// element (<c>c[i]</c>, <c>c[i + half]</c>), <see cref="RopeTableLayout.InterleavedPrefix"/> one per pair at
    /// <c>c[i]</c>, <see cref="RopeTableLayout.InterleavedPair"/> one per pair duplicated at <c>c[2i]</c>.</param>
    /// <param name="rotaryDim">Rotated width inside <c>D</c>; 0 rotates the whole vector.</param>
    /// <param name="dimOffset">First rotated element of each vector.</param>
    public static RotarySpec Create(StridedView vectors, StridedView table, RotaryPairing pairing, RopeTableLayout layout, int rotaryDim = 0, int dimOffset = 0)
    {
        if (vectors.Rank != 6 || table.Rank != 6)
            throw new ArgumentException($"Rotary base views have six axes; got vectors {vectors} and table {table}.", nameof(vectors));
        long width = vectors[5];
        long rotated = rotaryDim <= 0 || rotaryDim > width - dimOffset ? width - dimOffset : rotaryDim;
        if (rotated < 2 || (rotated & 1) != 0)
            throw new ArgumentException($"The rotated width must be even and at least 2; got {rotated} of {width} from offset {dimOffset}.", nameof(rotaryDim));
        long pairs = rotated / 2;
        StridedView x = vectors.Slice(5, dimOffset, rotated);
        x = pairing == RotaryPairing.Interleaved ? x.Split(5, pairs, 2) : x.Split(5, 2, pairs).Permute([0, 1, 2, 3, 4, 6, 5]);
        StridedView angles = layout switch
        {
            RopeTableLayout.RotateHalf => table.Slice(5, 0, rotated).Split(5, 2, pairs).Permute([0, 1, 2, 3, 4, 6, 5]),
            RopeTableLayout.InterleavedPrefix => table.Slice(5, 0, pairs).Broadcast(6, 2),
            _ => table.Slice(5, 0, pairs, 2).Broadcast(6, 2),
        };
        return new RotarySpec(x, angles, angles);
    }

    private static void RequireRank(TensorShape shape, int rank, string what)
    {
        if (shape.Rank != rank)
            throw new ArgumentException($"Expected {what}; got {shape}.", nameof(shape));
    }
}
