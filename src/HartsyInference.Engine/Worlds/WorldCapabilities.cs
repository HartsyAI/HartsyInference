using HartsyInference.Engine.Recipes;

namespace HartsyInference.Engine.Worlds;

/// <summary>What a world model accepts, declared on its catalog entry and checked by <see cref="WorldPlanResolver"/>
/// before any weight is loaded. The defaults describe the batch models that predate this record (Oasis, DIAMOND): any
/// backend, the native precision, 24 fps, 16 frames, no camera input, and completed frames delivered through
/// <see cref="Requests.IWorldSession.StreamAsync"/>.</summary>
public sealed record WorldCapabilities
{
    /// <summary>Backend kinds (<c>cpu</c>, <c>cuda</c>, <c>vulkan</c>) the model runs on; null accepts every backend.</summary>
    public IReadOnlyList<string>? Backends { get; init; }

    /// <summary>Precisions a request may name. The first is used when the request names none.</summary>
    public IReadOnlyList<string> Precisions { get; init; } = ["native"];

    /// <summary>Output frame rate when the request names none.</summary>
    public int DefaultFps { get; init; } = 24;

    /// <summary>Frame rates the model accepts; null accepts any positive rate.</summary>
    public IReadOnlyList<int>? Fps { get; init; }

    /// <summary>Frames to roll out when the request names none. Counts the seed frame.</summary>
    public int DefaultFrames { get; init; } = 16;

    /// <summary>Fewest frames a rollout may ask for. Counts the seed frame.</summary>
    public int MinFrames { get; init; } = 2;

    /// <summary>Valid frame counts; null accepts any count of at least <see cref="MinFrames"/>.</summary>
    public FrameGrid? Frames { get; init; }

    /// <summary>Denoising steps per frame when the request names none. Zero means the pipeline fixes its own count and
    /// any requested steps are ignored.</summary>
    public int DefaultSteps { get; init; }

    /// <summary>Camera inputs the model accepts. Empty means the model takes no camera path.</summary>
    public IReadOnlyList<WorldCameraMode> CameraModes { get; init; } = [];

    /// <summary>Expands <see cref="WorldCameraMode.ActionScript"/> camera input; required when <see cref="CameraModes"/> lists it.</summary>
    public IWorldCameraScript? CameraScript { get; init; }

    /// <summary>Whether a camera path is mandatory.</summary>
    public bool CameraRequired { get; init; }

    /// <summary>Whether pinhole intrinsics are accepted with the camera path.</summary>
    public WorldIntrinsicsPolicy Intrinsics { get; init; } = WorldIntrinsicsPolicy.NotAccepted;

    /// <summary>Lowest accepted horizontal and vertical field of view, in degrees, for typed intrinsics.</summary>
    public double MinFovDegrees { get; init; } = 25;

    /// <summary>Highest accepted horizontal and vertical field of view, in degrees, for typed intrinsics.</summary>
    public double MaxFovDegrees { get; init; } = 120;

    /// <summary>Sampler names the model accepts. Empty means it takes no sampler choice.</summary>
    public IReadOnlyList<string> Samplers { get; init; } = [];

    /// <summary>Whether the model takes a classifier-free guidance scale.</summary>
    public bool SupportsGuidance { get; init; }

    /// <summary>Whether the model takes a flow-matching shift.</summary>
    public bool SupportsFlowShift { get; init; }

    /// <summary>Whether the model has a refiner stage.</summary>
    public bool SupportsRefiner { get; init; }

    /// <summary>Whether the model yields partial frames while one is still denoising. Batch models cannot; they deliver
    /// each rollout's frames through <see cref="Requests.IWorldSession.StreamAsync"/> once it completes.</summary>
    public bool SupportsProgressivePreview { get; init; }

    /// <summary>Whether a request may ask for sequential classifier-free guidance: the conditional and unconditional passes
    /// run as two calls per step instead of one batch-2 call. No model declares it until its pipeline implements the path.</summary>
    public bool SupportsSequentialCfg { get; init; }

    /// <summary>Chunk-by-chunk streaming generation with live actions, when the model has it; null for a model that only
    /// renders whole rollouts.</summary>
    public WorldStreamingCapabilities? Streaming { get; init; }
}
