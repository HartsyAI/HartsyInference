namespace HartsyInference.ThreeD.Models.UniRig;

/// <summary>A generated bone hierarchy in the normalized [-1, 1] cube the model works in.</summary>
public sealed record UniRigSkeleton
{
    /// <summary>Joint head positions, length <c>3 * JointCount</c>.</summary>
    public required float[] Joints { get; init; }

    /// <summary>Bone tail positions, length <c>3 * JointCount</c>.</summary>
    public required float[] Tails { get; init; }

    /// <summary>Parent joint index per joint, <c>-1</c> for the root. A parent always precedes its child.</summary>
    public required int[] Parents { get; init; }

    /// <summary>Bone names (<c>bone_0</c> …) in joint order.</summary>
    public required string[] Names { get; init; }

    /// <summary>Part names in token order; <c>null</c> entries are spring tokens.</summary>
    public required string?[] Parts { get; init; }

    /// <summary>Number of joints.</summary>
    public int JointCount => Parents.Length;
}
