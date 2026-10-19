using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>adaLN shift/scale/gate that differ per <see cref="TokenSegment"/> group: one F32 table row per group, gathered
/// per token by <see cref="IBackend.AffineBroadcastRowIndexed"/> and <see cref="IBackend.GatedResidualRowIndexed"/>
/// through the layout's row index instead of materializing <c>[S, D]</c> rows. Used where reference tokens take a
/// different timestep's modulation than the live tokens. Operands are F32: Vulkan serves the row-indexed ops in F32 only.
/// The layout is borrowed; the tables are owned.</summary>
public sealed class SegmentModulation : IDisposable
{
    private readonly TokenSegmentLayout _layout;
    private int _disposed;

    private SegmentModulation(Tensor shift, Tensor scale, Tensor gate, TokenSegmentLayout layout)
    {
        Shift = shift;
        Scale = scale;
        Gate = gate;
        _layout = layout;
    }

    /// <summary>Shift table <c>[groups, D]</c>.</summary>
    public Tensor Shift { get; }

    /// <summary>Raw scale table <c>[groups, D]</c>; the <c>1 +</c> is applied by the consuming op.</summary>
    public Tensor Scale { get; }

    /// <summary>Residual gate table <c>[groups, D]</c>.</summary>
    public Tensor Gate { get; }

    /// <summary>Stacks one <c>(shift, scale, gate)</c> triple of F32 <c>[1, D]</c> rows per group of
    /// <paramref name="layout"/>. The rows stay owned by the caller.</summary>
    public static SegmentModulation Build(IBackend backend, IReadOnlyList<(Tensor Shift, Tensor Scale, Tensor Gate)> groupRows,
        TokenSegmentLayout layout)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(groupRows);
        ArgumentNullException.ThrowIfNull(layout);
        if (groupRows.Count != layout.GroupCount)
            throw new ArgumentException($"{groupRows.Count} modulation groups for a layout with {layout.GroupCount}.", nameof(groupRows));
        int dim = (int)groupRows[0].Shift.Shape[groupRows[0].Shift.Shape.Rank - 1];
        TensorShape tableShape = new(groupRows.Count, dim);
        Tensor[] parts = new Tensor[groupRows.Count];
        Tensor shift = Stack(backend, groupRows, parts, tableShape, 0);
        Tensor scale = Stack(backend, groupRows, parts, tableShape, 1);
        Tensor gate = Stack(backend, groupRows, parts, tableShape, 2);
        return new SegmentModulation(shift, scale, gate, layout);
    }

    /// <summary>Returns <c>normed·(1 + scale[g]) + shift[g]</c> per token, F32.</summary>
    public Tensor Affine(IBackend backend, Tensor normed)
    {
        ArgumentNullException.ThrowIfNull(backend);
        RequireRows(normed);
        Tensor output = new(normed.Shape, DType.F32);
        backend.AffineBroadcastRowIndexed(output, normed, Scale, Shift, _layout.RowIndex);
        return output;
    }

    /// <summary>Returns <c>residual + gate[g]·value</c> per token, F32.</summary>
    public Tensor GatedResidual(IBackend backend, Tensor residual, Tensor value)
    {
        ArgumentNullException.ThrowIfNull(backend);
        RequireRows(residual);
        Tensor output = new(residual.Shape, DType.F32);
        backend.GatedResidualRowIndexed(output, residual, value, Gate, _layout.RowIndex);
        return output;
    }

    private void RequireRows(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);
        long rows = x.ElementCount / x.Shape[x.Shape.Rank - 1];
        if (rows != _layout.TotalLength)
            throw new ArgumentException($"A {_layout.TotalLength}-token segment layout cannot modulate {rows} rows ({x.Shape}).", nameof(x));
    }

    private static Tensor Stack(IBackend backend, IReadOnlyList<(Tensor Shift, Tensor Scale, Tensor Gate)> rows, Tensor[] parts,
        TensorShape shape, int which)
    {
        for (int i = 0; i < rows.Count; i++) parts[i] = which == 0 ? rows[i].Shift : which == 1 ? rows[i].Scale : rows[i].Gate;
        Tensor table = new(shape, DType.F32);
        backend.Concat(table, parts, 0);
        return table;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Shift.Dispose();
        Scale.Dispose();
        Gate.Dispose();
    }
}
