namespace HartsyInference.Core.Rope;

/// <summary>Values written to table entries the layout does not read.</summary>
public enum RopeTablePadding
{
    /// <summary>Unread entries are identity rotations: <c>cos = 1</c>, <c>sin = 0</c>.</summary>
    Identity,

    /// <summary>Unread entries are zero, matching tables that were built with zero padding.</summary>
    Zero,
}
