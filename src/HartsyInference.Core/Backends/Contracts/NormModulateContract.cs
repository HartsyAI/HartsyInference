using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Contracts;

/// <summary>The exact contract of <see cref="IBackend.NormModulate"/>: F32, F16 or BF16 activations of equal element count,
/// F32 vectors of the row width, a positive epsilon for a norm, a modulation along the last axis only whose vectors cover
/// every modulation row the rows read, an I32 index of one entry per row when given, and no storage shared between the
/// output and any operand. Every implementation validates through here.</summary>
public static class NormModulateContract
{
    public static NormModulateGeometry Validate(in NormModulateCall call)
    {
        ArgumentNullException.ThrowIfNull(call.Output);
        ArgumentNullException.ThrowIfNull(call.Input);
        RequireActivation(call.Input.DType, "input");
        RequireActivation(call.Output.DType, "output");
        if (call.Output.ElementCount != call.Input.ElementCount)
            throw new ArgumentException($"NormModulate output {call.Output.Shape} and input {call.Input.Shape} differ in element count.", nameof(call));
        if (call.Kind != NormKind.None && !(call.Eps > 0f))
            throw new ArgumentException($"A {call.Kind} needs a positive epsilon; got {call.Eps}.", nameof(call));
        if (!Enum.IsDefined(call.Kind))
            throw new ArgumentOutOfRangeException(nameof(call), call.Kind, "Unknown norm kind.");
        int width;
        long rows, spatial;
        if (call.Axis == NormAxis.LastDim)
        {
            width = (int)call.Input.Shape[call.Input.Shape.Rank - 1];
            rows = call.Input.ElementCount / width;
            spatial = 1;
        }
        else if (call.Axis == NormAxis.Channel)
        {
            if (call.Input.Shape.Rank < 3)
                throw new ArgumentException($"A channel-axis row needs an input [N, C, ...]; got {call.Input.Shape}.", nameof(call));
            width = (int)call.Input.Shape[1];
            spatial = call.Input.ElementCount / (call.Input.Shape[0] * width);
            rows = call.Input.Shape[0] * spatial;
        }
        else
        {
            throw new ArgumentOutOfRangeException(nameof(call), call.Axis, "Unknown norm axis.");
        }
        if (width <= 0)
            throw new ArgumentException($"A row needs at least one element; got {call.Input.Shape}.", nameof(call));
        RequireVector(call.Weight, width, "weight");
        RequireVector(call.Bias, width, "bias");
        ModulationSpec mod = call.Modulation;
        // An elementwise call may update its input in place: every element is read before it is written. Every call, modulated or not,
        // runs this check before it returns: an unmodulated norm writes rows it may not have read yet just as a modulated one does.
        bool inPlace = call.Kind == NormKind.None && ReferenceEquals(call.Input, call.Output);
        foreach (Tensor? operand in new[] { call.Input, call.Weight, call.Bias, mod.Scale, mod.Shift, mod.RowIndex })
        {
            if (operand is null || (inPlace && ReferenceEquals(operand, call.Input))) continue;
            if (call.Output.HasOverlappingHostStorageWithoutSync(operand))
                throw new ArgumentException("NormModulate output storage must not overlap an operand.", nameof(call));
        }
        if (mod.IsNone)
        {
            if (mod.RowIndex is not null)
                throw new ArgumentException("A row index applies only with a scale or shift.", nameof(call));
            return new NormModulateGeometry(rows, width, spatial, 0, width, 0, 0);
        }
        if (call.Axis != NormAxis.LastDim)
            throw new NotSupportedException("A modulation applies along the last axis only.");
        if (!Enum.IsDefined(mod.Mode))
            throw new ArgumentOutOfRangeException(nameof(call), mod.Mode, "Unknown scale mode.");
        if (mod.RowStride < width || mod.ScaleOffset < 0 || mod.ShiftOffset < 0 || mod.RowsPerMod < 0)
            throw new ArgumentException($"Modulation rows are at least {width} apart with non-negative offsets; got stride {mod.RowStride}, offsets {mod.ScaleOffset}/{mod.ShiftOffset}, rows per modulation {mod.RowsPerMod}.", nameof(call));
        long modRows;
        if (mod.RowIndex is not null)
        {
            if (mod.RowIndex.DType != DType.I32 || mod.RowIndex.ElementCount < rows)
                throw new ArgumentException($"The row index is I32 with one entry per row ({rows}); got {mod.RowIndex.DType} {mod.RowIndex.Shape}.", nameof(call));
            modRows = ModulationRowsFor(mod, width);
        }
        else
        {
            modRows = rows == 0 ? 0 : (mod.RowsPerMod == 0 ? 0 : (rows - 1) / mod.RowsPerMod) + 1;
        }
        RequireModulationRows(mod.Scale, mod.ScaleOffset, mod.RowStride, width, modRows, "scale");
        RequireModulationRows(mod.Shift, mod.ShiftOffset, mod.RowStride, width, modRows, "shift");
        return new NormModulateGeometry(rows, width, spatial, mod.RowIndex is null ? mod.RowsPerMod : 1, mod.RowStride, mod.ScaleOffset, mod.ShiftOffset);
    }

    /// <summary>The modulation rows a row-indexed spec can address: the smaller of the scale's and shift's row counts.</summary>
    public static long ModulationRowsFor(in ModulationSpec mod, int width)
    {
        long rows = long.MaxValue;
        if (mod.Scale is not null) rows = Math.Min(rows, RowsAvailable(mod.Scale, mod.ScaleOffset, mod.RowStride, width));
        if (mod.Shift is not null) rows = Math.Min(rows, RowsAvailable(mod.Shift, mod.ShiftOffset, mod.RowStride, width));
        return rows;
    }

    private static long RowsAvailable(Tensor table, long offset, long stride, int width)
        => table.ElementCount < offset + width ? 0 : (table.ElementCount - offset - width) / stride + 1;

    private static void RequireActivation(DType dtype, string name)
    {
        if (dtype != DType.F32 && dtype != DType.F16 && dtype != DType.BF16)
            throw new NotSupportedException($"NormModulate takes an F32, F16 or BF16 {name}; got {dtype}.");
    }

    private static void RequireVector(Tensor? vector, int width, string name)
    {
        if (vector is null) return;
        if (vector.DType != DType.F32 || vector.ElementCount != width)
            throw new ArgumentException($"The {name} is F32 with {width} elements; got {vector.DType} {vector.Shape}.", name);
    }

    private static void RequireModulationRows(Tensor? table, long offset, long stride, int width, long modRows, string name)
    {
        if (table is null) return;
        if (table.DType != DType.F32)
            throw new ArgumentException($"The modulation {name} is F32; got {table.DType}.", name);
        if (modRows > 0 && RowsAvailable(table, offset, stride, width) < modRows)
            throw new ArgumentException($"The modulation {name} {table.Shape} holds fewer than the {modRows} rows of {width} (stride {stride}, offset {offset}) the call reads.", name);
    }
}
