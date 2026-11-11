namespace HartsyInference.ThreeD.Io;

/// <summary>One bone of a rest-pose skeleton for <see cref="SkinnedGlbWriter"/>.</summary>
/// <param name="Name">Node name in the file.</param>
/// <param name="Parent">Index of the parent bone, <c>-1</c> for a root; a parent precedes its children.</param>
/// <param name="Head">World-space head position (x, y, z); the bone's joint origin.</param>
/// <param name="Tail">World-space tail position, stored in the node's <c>extras</c> only.</param>
public readonly record struct SkinnedBone(string Name, int Parent, (float X, float Y, float Z) Head, (float X, float Y, float Z) Tail);
