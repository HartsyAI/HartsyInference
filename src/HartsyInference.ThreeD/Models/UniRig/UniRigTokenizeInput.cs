namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>A skeleton laid out for <see cref="UniRigTokenizer.Tokenize"/>: heads, parent positions and branch flags.</summary>
public sealed record UniRigTokenizeInput
{
    /// <summary>Joint heads in token order, length <c>3 * JointCount</c>.</summary>
    public required float[] Joints { get; init; }

    /// <summary>Position of each joint's parent, length <c>3 * JointCount</c>; only read for branch joints.</summary>
    public required float[] ParentJoints { get; init; }

    /// <summary>True when the joint does not follow its parent in the sequence and so carries a branch marker. Always false for the root.</summary>
    public required bool[] Branch { get; init; }

    /// <summary>Class name placed after the BOS token; an unknown or null name emits the no-class token.</summary>
    public string? Cls { get; init; }

    /// <summary>Part token before joint <c>i</c>: a part name, or <c>null</c> for a spring token.</summary>
    public IReadOnlyDictionary<int, string?> PartsBias { get; init; } = new Dictionary<int, string?>();
}
