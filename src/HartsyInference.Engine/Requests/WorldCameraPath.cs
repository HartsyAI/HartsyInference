namespace HartsyInference.Engine.Requests;

/// <summary>Typed camera input: one camera-to-world pose per frame, plus the intrinsics the poses are projected with.</summary>
/// <remarks>Each pose is 16 row-major values of a 4x4 camera-to-world matrix in OpenCV convention: +X right, +Y down,
/// +Z forward. Pose <c>i</c> belongs to frame <c>i</c>, and frame 0 is the seed image.</remarks>
public sealed record WorldCameraPath
{
    /// <summary>The per-frame poses, each 16 row-major values.</summary>
    public required IReadOnlyList<float[]> Poses { get; init; }

    /// <summary>Intrinsics for the poses; null when the model does not take them or they were not supplied.</summary>
    public WorldIntrinsics? Intrinsics { get; init; }
}
