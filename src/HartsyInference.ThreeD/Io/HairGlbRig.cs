namespace HartsyInference.ThreeD.Io;

/// <summary>
/// Skin of one node: for every vertex, four joint indices and four weights (glTF JOINTS_0 and WEIGHTS_0). Unused slots have
/// weight 0. Every vertex's weights must sum to 1.
/// </summary>
public sealed record HairGlbSkin(ushort[] JointIndices, float[] Weights);

/// <summary>
/// One joint of the rig: a root-level glTF node with its rest translation (xyz) and rest rotation (quaternion xyzw), and its
/// inverse bind matrix (16 values, column-major).
/// </summary>
public sealed record HairGlbJoint(string Name, float[] Translation, float[] Rotation, float[] InverseBindMatrix);

/// <summary>Per-frame translation (3 values per frame) and rotation (quaternion xyzw, 4 values per frame) of one joint.</summary>
public sealed record HairGlbChannel(float[] Translations, float[] Rotations);

/// <summary>A named clip: one time per key frame (seconds) and one channel per joint, in joint order.</summary>
public sealed record HairGlbAnimation(string Name, float[] Times, IReadOnlyList<HairGlbChannel> Channels);

/// <summary>The joints of a skinned export and the clips that drive them.</summary>
public sealed record HairGlbRig(IReadOnlyList<HairGlbJoint> Joints, IReadOnlyList<HairGlbAnimation> Animations);
