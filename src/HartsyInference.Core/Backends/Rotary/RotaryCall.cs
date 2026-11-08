using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Rotary;

/// <summary>One <see cref="IBackend.Rotary"/> invocation: the tensors a <see cref="RotarySpec"/> addresses. The rotation
/// is in place on <see cref="X"/>. The factories cover the layouts the models use; a model with another geometry
/// builds its spec with <see cref="RotaryViews"/> and the constructor.</summary>
/// <param name="X">The activation, F32, F16 or BF16.</param>
/// <param name="Cos">The cosine table, F32.</param>
/// <param name="Sin">The sine table, F32.</param>
/// <param name="Spec">Which elements pair up and which entries they read.</param>
public readonly record struct RotaryCall(Tensor X, Tensor Cos, Tensor Sin, RotarySpec Spec)
{
    /// <summary>Token-major <paramref name="x"/> <c>[B, L, H, D]</c> (or <c>[B, L, D]</c> as one head) with tables <c>[L, E]</c> or <c>[B, L, E]</c> shared by every head.</summary>
    public static RotaryCall TokenMajor(Tensor x, Tensor cos, Tensor sin, RotaryPairing pairing, RopeTableLayout layout, int rotaryDim = 0, int dimOffset = 0)
    {
        TensorShape shape = x.Shape.Rank == 3 ? new TensorShape(x.Shape[0], x.Shape[1], 1, x.Shape[2]) : x.Shape;
        StridedView vectors = RotaryViews.TokenMajor(shape);
        return new(x, cos, sin, RotaryViews.Create(vectors, RotaryViews.SharedTable(cos.Shape, shape[0], shape[2]), pairing, layout, rotaryDim, dimOffset));
    }

    /// <summary>Head-major <paramref name="x"/> <c>[B, H, L, D]</c> with tables <c>[L, E]</c> or <c>[B, L, E]</c> shared by every head.</summary>
    public static RotaryCall HeadMajor(Tensor x, Tensor cos, Tensor sin, RotaryPairing pairing, RopeTableLayout layout, int rotaryDim = 0, int dimOffset = 0)
        => new(x, cos, sin, RotaryViews.Create(RotaryViews.HeadMajor(x.Shape), RotaryViews.SharedTable(cos.Shape, x.Shape[0], x.Shape[1]), pairing, layout, rotaryDim, dimOffset));

    /// <summary>Rows <paramref name="x"/> <c>[S, heads·D]</c> (any shape with that element count) with tables <c>[S, E]</c> shared by every head.</summary>
    public static RotaryCall Rows(Tensor x, Tensor cos, Tensor sin, long tokens, int heads, int headDim, RotaryPairing pairing, RopeTableLayout layout, int rotaryDim = 0)
        => new(x, cos, sin, RotaryViews.Create(RotaryViews.Rows(tokens, heads, headDim), RotaryViews.SharedTable(cos.Shape, 1, heads).Slice(2, 0, tokens), pairing, layout, rotaryDim));

    /// <summary>Rows <paramref name="x"/> <c>[S, heads·D]</c> with a table per head, <c>[heads, S, E]</c>.</summary>
    public static RotaryCall RowsPerHead(Tensor x, Tensor cos, Tensor sin, long tokens, int heads, int headDim, RotaryPairing pairing, RopeTableLayout layout, int rotaryDim = 0)
        => new(x, cos, sin, RotaryViews.Create(RotaryViews.Rows(tokens, heads, headDim), RotaryViews.PerHeadTable(cos.Shape, 1), pairing, layout, rotaryDim));

    /// <summary>Rows <paramref name="x"/> <c>[S, heads·D]</c> rotated by halves with one angle per pair, the angles packed per head in <c>[S, heads·D/2]</c> rows (LTX-2 split).</summary>
    public static RotaryCall SplitRows(Tensor x, Tensor cos, Tensor sin, long tokens, int heads, int headDim)
        => new(x, cos, sin, RotaryViews.SplitRows(tokens, heads, headDim));

    /// <summary>A decode step's <paramref name="x"/> of <c>heads × D</c> elements (<c>[1, H, 1, D]</c> or <c>[1, 1, H, D]</c>) rotated by the row of the
    /// full <c>[maxPos, D]</c> tables that the device position buffer's query offset names.</summary>
    public static RotaryCall DecodeStep(Tensor x, Tensor cosTable, Tensor sinTable, int rotaryDim, bool interleaved, ulong devicePos)
    {
        if (devicePos == 0)
            throw new ArgumentException("A decode-step rotation reads its row from a device position buffer.", nameof(devicePos));
        int headDim = (int)x.Shape[x.Shape.Rank - 1];
        long heads = x.ElementCount / headDim;
        long entries = cosTable.Shape[cosTable.Shape.Rank - 1];
        StridedView vectors = RotaryViews.Explicit([1, heads, 1, 1, 1, headDim], [0, headDim, 0, 0, 0, 1]);
        StridedView table = RotaryViews.Explicit([1, heads, 1, 1, 1, entries], [0, 0, 0, 0, 0, 1]);
        // The interleaved decode table holds one angle per pair of the whole head; the split-half one holds an entry per rotated element.
        RotarySpec spec = interleaved
            ? RotaryViews.Create(vectors, table, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPrefix)
            : RotaryViews.Create(vectors, table, RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf, rotaryDim);
        return new(x, cosTable, sinTable, spec.AtDevicePosition(devicePos, 1, entries));
    }
}
