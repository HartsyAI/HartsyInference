using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>The shift/scale/gate a FLUX.3 block applies. Uniform modulation holds <c>[1, D]</c> rows; span
/// modulation holds one table row per conditioning mode plus an I32 <c>[S]</c> row index naming each token's mode,
/// consumed by <see cref="IBackend.AffineBroadcastRowIndexed"/> without materializing per-token rows.</summary>
public sealed class Flux3Modulation : IDisposable
{
    private int _disposed;

    private Flux3Modulation(Tensor shift, Tensor scale, Tensor gate, Tensor? rowIndex)
    {
        Shift = shift;
        Scale = scale;
        Gate = gate;
        RowIndex = rowIndex;
    }

    /// <summary>Shift row(s).</summary>
    public Tensor Shift { get; }

    /// <summary>Raw scale row(s); the <c>1 +</c> is applied by the consuming op.</summary>
    public Tensor Scale { get; }

    /// <summary>Residual gate row(s).</summary>
    public Tensor Gate { get; }

    /// <summary>Per-token table row, or null for uniform modulation.</summary>
    public Tensor? RowIndex { get; }

    /// <summary>Takes ownership of three <c>[1, D]</c> rows applied to every token.</summary>
    public static Flux3Modulation Uniform(Tensor shift, Tensor scale, Tensor gate) => new(shift, scale, gate, null);

    /// <summary>Stacks one <c>(shift, scale, gate)</c> triple per span into tables and indexes every token of span
    /// <c>i</c> to row <c>i</c>. The source rows stay owned by the caller.</summary>
    public static unsafe Flux3Modulation Spans(IBackend backend, IReadOnlyList<(Tensor Shift, Tensor Scale, Tensor Gate)> rows,
        IReadOnlyList<int> spanLengths)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(rows);
        ArgumentNullException.ThrowIfNull(spanLengths);
        if (rows.Count == 0 || rows.Count != spanLengths.Count)
            throw new ArgumentException($"{rows.Count} modulation rows for {spanLengths.Count} spans.", nameof(spanLengths));
        int dim = (int)rows[0].Shift.Shape[rows[0].Shift.Shape.Rank - 1];
        TensorShape tableShape = new TensorShape(rows.Count, dim);
        Tensor shift = Stack(backend, rows, tableShape, static r => r.Shift);
        Tensor scale = Stack(backend, rows, tableShape, static r => r.Scale);
        Tensor gate = Stack(backend, rows, tableShape, static r => r.Gate);
        int total = 0;
        foreach (int length in spanLengths) total += length;
        Tensor rowIndex = new Tensor(new TensorShape(total), DType.I32);
        int* p = (int*)rowIndex.DataPointer;
        int at = 0;
        for (int span = 0; span < spanLengths.Count; span++)
        {
            for (int i = 0; i < spanLengths[span]; i++) p[at++] = span;
        }
        return new Flux3Modulation(shift, scale, gate, rowIndex);
    }

    /// <summary><c>out = (1 + scale)·LayerNorm(x) + shift</c> with this modulation.</summary>
    public Tensor NormModulate(IBackend backend, Tensor x, float eps)
    {
        Tensor output = new Tensor(x.Shape, x.DType);
        if (RowIndex is null)
        {
            backend.LayerNormModulate(output, x, Scale, Shift, eps);
            return output;
        }
        Tensor normed = new Tensor(x.Shape, x.DType);
        backend.LayerNormNoAffine(normed, x, eps);
        backend.AffineBroadcastRowIndexed(output, normed, Scale, Shift, RowIndex);
        normed.Dispose();
        return output;
    }

    /// <summary><c>out = residual + gate·value</c> with this modulation's gate.</summary>
    public Tensor GatedResidual(IBackend backend, Tensor residual, Tensor value)
    {
        Tensor output = new Tensor(residual.Shape, residual.DType);
        if (RowIndex is null)
            backend.GatedResidualLastDim(output, residual, value, Gate);
        else
            backend.GatedResidualRowIndexed(output, residual, value, Gate, RowIndex);
        return output;
    }

    private static Tensor Stack(IBackend backend, IReadOnlyList<(Tensor Shift, Tensor Scale, Tensor Gate)> rows,
        TensorShape shape, Func<(Tensor Shift, Tensor Scale, Tensor Gate), Tensor> select)
    {
        Tensor[] parts = new Tensor[rows.Count];
        for (int i = 0; i < rows.Count; i++) parts[i] = select(rows[i]);
        Tensor table = new Tensor(shape, DType.F32);
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
        RowIndex?.Dispose();
    }
}
