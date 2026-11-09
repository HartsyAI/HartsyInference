namespace HartsyInference.Core.Backends.Gating;

/// <summary>One <see cref="IBackend.GroupScale"/>: every row of the activation is cut into groups of <see cref="GroupWidth"/>
/// elements and group <c>g</c> of row <c>r</c> is scaled by <c>f(gates[GateOffset + r · GateStride + g])</c>, written over the
/// output or added to it. A group width of 0 is the whole row.</summary>
/// <param name="GroupWidth">Elements per gate within a row; 0 for one gate per row.</param>
/// <param name="GateStride">Gate elements between consecutive rows.</param>
/// <param name="GateOffset">The gate element of row 0, group 0.</param>
/// <param name="Function">What the gate passes through.</param>
/// <param name="Accumulate">Add the scaled value to the output instead of writing it; a zero gate leaves the row untouched.</param>
public readonly record struct GroupScaleSpec(int GroupWidth, long GateStride, long GateOffset, GateFunction Function, bool Accumulate)
{
    /// <summary>One gate per row from a <c>[rows]</c> vector (token masking).</summary>
    public static GroupScaleSpec PerRow(GateFunction function = GateFunction.Identity) => new(0, 1, 0, function, false);

    /// <summary>One gate per group of <paramref name="groupWidth"/> elements from <c>[rows, groupsPerRow]</c> gates (a per-head gate).</summary>
    public static GroupScaleSpec PerGroup(int groupWidth, int groupsPerRow, GateFunction function) => new(groupWidth, groupsPerRow, 0, function, false);

    /// <summary>One gate per row read from column <paramref name="column"/> of a <c>[rows, columns]</c> table (one expert of a dense routing table).</summary>
    public static GroupScaleSpec Column(long columns, long column, bool accumulate) => new(0, columns, column, GateFunction.Identity, accumulate);
}
