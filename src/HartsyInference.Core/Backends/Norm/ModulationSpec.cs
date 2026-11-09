using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Norm;

/// <summary>The modulation of a <see cref="NormModulateCall"/>: which scale and shift vector each row reads and how the
/// scale multiplies. Row <c>r</c> reads modulation row <c>m = RowIndex[r]</c> when an index is given, else
/// <c>r / RowsPerMod</c> (every row reads row 0 when <see cref="RowsPerMod"/> is 0); the vectors start at
/// <c>ScaleOffset + m · RowStride</c> of <see cref="Scale"/> and <c>ShiftOffset + m · RowStride</c> of <see cref="Shift"/>, so a
/// packed <c>[mods, k · width]</c> modulation output serves without being cut. A null scale or shift drops its term.</summary>
/// <param name="Scale">The scale vectors, F32, or null.</param>
/// <param name="Shift">The shift vectors, F32, or null; may be the same tensor as <paramref name="Scale"/>.</param>
/// <param name="Mode">How the scale multiplies.</param>
/// <param name="RowsPerMod">Consecutive rows that share one modulation row; 0 shares row 0 across every row.</param>
/// <param name="RowStride">Elements between consecutive modulation rows.</param>
/// <param name="ScaleOffset">The first element of modulation row 0's scale vector.</param>
/// <param name="ShiftOffset">The first element of modulation row 0's shift vector.</param>
/// <param name="RowIndex">I32 <c>[rows]</c>, one modulation row per row, in place of <paramref name="RowsPerMod"/>.</param>
public readonly record struct ModulationSpec(Tensor? Scale, Tensor? Shift, ScaleMode Mode, long RowsPerMod, long RowStride, long ScaleOffset, long ShiftOffset, Tensor? RowIndex)
{
    /// <summary>No modulation.</summary>
    public static ModulationSpec None => default;

    /// <summary>True when neither a scale nor a shift is applied.</summary>
    public bool IsNone => Scale is null && Shift is null;

    /// <summary>One scale and shift vector of <paramref name="width"/> for every row.</summary>
    public static ModulationSpec Shared(Tensor? scale, Tensor? shift, int width, ScaleMode mode) => new(scale, shift, mode, 0, width, 0, 0, null);

    /// <summary>Vectors <c>[mods, width]</c>, each shared by <paramref name="rowsPerMod"/> consecutive rows (a per-batch modulation over a sequence).</summary>
    public static ModulationSpec PerBatch(Tensor? scale, Tensor? shift, int width, long rowsPerMod, ScaleMode mode) => new(scale, shift, mode, rowsPerMod, width, 0, 0, null);

    /// <summary>Vectors <c>[mods, width]</c> gathered per row through <paramref name="rowIndex"/>.</summary>
    public static ModulationSpec RowIndexed(Tensor? scale, Tensor? shift, int width, Tensor rowIndex, ScaleMode mode) => new(scale, shift, mode, 1, width, 0, 0, rowIndex);

    /// <summary>Vectors <c>[mods, width]</c> broadcast over <paramref name="activation"/>'s rows, each shared by an equal run of consecutive
    /// rows: <c>[D]</c> over every row, <c>[B, D]</c> over each batch of a <c>[B, S, D]</c> activation. The row count must divide evenly.</summary>
    public static ModulationSpec Broadcast(Tensor activation, Tensor? scale, Tensor? shift, ScaleMode mode)
    {
        int width = (int)activation.Shape[activation.Shape.Rank - 1];
        Tensor table = scale ?? shift ?? throw new ArgumentException("A broadcast modulation needs a scale or a shift.", nameof(scale));
        long rows = activation.ElementCount / width;
        long modRows = table.ElementCount / width;
        if (modRows <= 0 || table.ElementCount != modRows * width || rows % modRows != 0)
            throw new ArgumentException($"A modulation {table.Shape} is [D] or whole rows of {width} that divide the {rows} rows of {activation.Shape}.", nameof(scale));
        return PerBatch(scale, shift, width, rows / modRows, mode);
    }

    /// <summary>Vectors <c>[mods, width]</c> gathered per row of <paramref name="activation"/> through <paramref name="rowIndex"/>.</summary>
    public static ModulationSpec RowIndexed(Tensor activation, Tensor? scale, Tensor? shift, Tensor rowIndex, ScaleMode mode)
        => RowIndexed(scale, shift, (int)activation.Shape[activation.Shape.Rank - 1], rowIndex, mode);

    /// <summary>Scale and shift cut from one packed modulation output <c>[mods, rowStride]</c> at the given offsets, each row shared by <paramref name="rowsPerMod"/> rows.</summary>
    public static ModulationSpec Packed(Tensor modulation, long rowStride, long scaleOffset, long shiftOffset, long rowsPerMod, ScaleMode mode)
        => new(modulation, modulation, mode, rowsPerMod, rowStride, scaleOffset, shiftOffset, null);
}
