using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The exact contracts of the row-index ops: F32 rows addressed by an I32 index. The index is bounds-checked (and,
/// for the scatter-add, checked for repeats) when its host bytes are current; an index that lives on the device is the caller's
/// to keep in range and, for the scatter-add, free of repeats — the kernels add without atomics.</summary>
public static class RowIndexContract
{
    /// <summary><see cref="IBackend.GatherRows"/>: <c>output[m] = input[indices[m]]</c>. Repeated indices are valid.</summary>
    public static RowIndexGeometry ValidateGather(Tensor output, Tensor input, Tensor indices)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(indices);
        RequireDtypes("GatherRows", output, input, indices);
        int width = Width("GatherRows", input);
        long rows = Count("GatherRows", indices);
        if (output.ElementCount != rows * width)
            throw new ArgumentException($"GatherRows output {output.Shape} must hold {rows} rows of width {width}; it has {output.ElementCount} elements.", nameof(output));
        if (output.HasOverlappingHostStorageWithoutSync(input))
            throw new ArgumentException("GatherRows output storage must not overlap the input.", nameof(output));
        long inputRows = input.ElementCount / width;
        CheckRange("GatherRows", indices, inputRows, unique: false);
        return new RowIndexGeometry(rows, width, inputRows);
    }

    /// <summary><see cref="IBackend.ScatterAddRows"/>: <c>output[indices[m]] += weights[m] · input[m]</c>, in place on the output.
    /// Each destination row is visited at most once.</summary>
    public static RowIndexGeometry ValidateScatterAdd(Tensor output, Tensor input, Tensor indices, Tensor? weights)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(indices);
        RequireDtypes("ScatterAddRows", output, input, indices);
        int width = Width("ScatterAddRows", input);
        long rows = Count("ScatterAddRows", indices);
        if (input.ElementCount != rows * width)
            throw new ArgumentException($"ScatterAddRows input {input.Shape} must hold {rows} rows of width {width}; it has {input.ElementCount} elements.", nameof(input));
        if (output.ElementCount % width != 0)
            throw new ArgumentException($"ScatterAddRows output {output.Shape} is not a whole number of rows of width {width}.", nameof(output));
        if (weights is not null && (weights.DType != DType.F32 || weights.ElementCount != rows))
            throw new ArgumentException($"ScatterAddRows weights must be F32 with one entry per scattered row ({rows}); got {weights.DType} {weights.Shape}.", nameof(weights));
        if (output.HasOverlappingHostStorageWithoutSync(input))
            throw new ArgumentException("ScatterAddRows output storage must not overlap the input.", nameof(output));
        if (weights is not null && output.HasOverlappingHostStorageWithoutSync(weights))
            throw new ArgumentException("ScatterAddRows output storage must not overlap the weights: the scatter rewrites values the later rows still read.", nameof(output));
        long outputRows = output.ElementCount / width;
        CheckRange("ScatterAddRows", indices, outputRows, unique: true);
        return new RowIndexGeometry(rows, width, outputRows);
    }

    /// <summary><see cref="IBackend.GatherAddRows"/>: <c>h[r] += table[indices[r]]</c> for every row of <paramref name="h"/>, in place.</summary>
    public static RowIndexGeometry ValidateGatherAdd(Tensor h, Tensor table, Tensor indices)
    {
        ArgumentNullException.ThrowIfNull(h);
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(indices);
        RequireDtypes("GatherAddRows", h, table, indices);
        int width = Width("GatherAddRows", table);
        long rows = Count("GatherAddRows", indices);
        if (h.ElementCount != rows * width)
            throw new ArgumentException($"GatherAddRows h {h.Shape} must hold one row of width {width} per index ({rows}); it has {h.ElementCount} elements.", nameof(h));
        if (h.HasOverlappingHostStorageWithoutSync(table))
            throw new ArgumentException("GatherAddRows h storage must not overlap the table.", nameof(h));
        long tableRows = table.ElementCount / width;
        CheckRange("GatherAddRows", indices, tableRows, unique: false);
        return new RowIndexGeometry(rows, width, tableRows);
    }

    private static void RequireDtypes(string op, Tensor a, Tensor b, Tensor indices)
    {
        if (a.DType != DType.F32 || b.DType != DType.F32)
            throw new NotSupportedException($"{op} takes F32 rows; got {a.DType} and {b.DType}.");
        if (indices.DType != DType.I32)
            throw new NotSupportedException($"{op} takes an I32 index; got {indices.DType}.");
    }

    private static int Width(string op, Tensor rows)
    {
        if (rows.Shape.Rank < 1)
            throw new ArgumentException($"{op} rows need at least one dimension.", nameof(rows));
        long width = rows.Shape[rows.Shape.Rank - 1];
        if (width <= 0 || width > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(rows), width, $"{op} row width must be in [1, {int.MaxValue}].");
        return (int)width;
    }

    private static long Count(string op, Tensor indices)
    {
        if (indices.ElementCount > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(indices), indices.ElementCount, $"{op} addresses at most {int.MaxValue} rows.");
        return indices.ElementCount;
    }

    private static unsafe void CheckRange(string op, Tensor indices, long indexedRows, bool unique)
    {
        if (!indices.HostBytesCurrent) return;
        int count = (int)indices.ElementCount;
        int* p = (int*)indices.DataPointer;
        HashSet<int>? seen = unique && count > 1 ? new HashSet<int>(count) : null;
        for (int i = 0; i < count; i++)
        {
            int row = p[i];
            if (row < 0 || row >= indexedRows)
                throw new ArgumentOutOfRangeException(nameof(indices), row, $"{op} index at position {i} is outside [0, {indexedRows}).");
            if (seen is not null && !seen.Add(row))
                throw new ArgumentException($"{op} index names row {row} more than once; each destination row is visited at most once.", nameof(indices));
        }
    }
}
