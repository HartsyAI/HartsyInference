namespace HartsyInference.Engine.Requests;

/// <summary>Typed camera input: one camera-to-world pose per frame, or a script in the model's action language that expands to
/// them, plus the intrinsics the poses are projected with.</summary>
/// <remarks>Each pose is 16 row-major values of a 4x4 camera-to-world matrix in OpenCV convention: +X right, +Y down,
/// +Z forward. Pose <c>i</c> belongs to frame <c>i</c>, and frame 0 is the seed image. A path carries poses or a script, not
/// both; planning expands a script into poses.</remarks>
public sealed record WorldCameraPath
{
    /// <summary>The per-frame poses, each 16 row-major values; empty when <see cref="Script"/> carries the path.</summary>
    public IReadOnlyList<float[]> Poses { get; init; } = [];

    /// <summary>The path in the model's action language (for SANA-WM, <c>w-80,dw-40</c>); null for a pose path.</summary>
    public string? Script { get; init; }

    /// <summary>Intrinsics for every frame; null when the model does not take them, they were not supplied, or
    /// <see cref="FrameIntrinsics"/> carries them.</summary>
    public WorldIntrinsics? Intrinsics { get; init; }

    /// <summary>Per-frame intrinsics as supplied, fitted to the frame count by the model (truncated, or resampled over time);
    /// null when <see cref="Intrinsics"/> covers every frame.</summary>
    public IReadOnlyList<WorldIntrinsics>? FrameIntrinsics { get; init; }
}
