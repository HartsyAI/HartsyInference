using HartsyInference.Core.Memory.Vram;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Worlds;

/// <summary>The flat option set a world command passes in (CLI flags, or the same fields from any other transport), mapped
/// to a <see cref="WorldRequest"/>. A null field means "the model default", so nothing is filled in on the caller's behalf.</summary>
public sealed record WorldRequestOptions
{
    /// <summary>The prompt the world is seeded with.</summary>
    public string? Prompt { get; init; }

    /// <summary>Frames to roll out, counting the seed frame.</summary>
    public int? Frames { get; init; }

    /// <summary>Output frame rate.</summary>
    public int? Fps { get; init; }

    /// <summary>Denoising steps per frame.</summary>
    public int? Steps { get; init; }

    /// <summary>Classifier-free guidance scale.</summary>
    public double? Guidance { get; init; }

    /// <summary>Flow-matching schedule shift.</summary>
    public double? FlowShift { get; init; }

    /// <summary>RNG seed; negative draws one.</summary>
    public long Seed { get; init; } = -1;

    /// <summary>Path to a camera path file (<c>.json</c> or <c>.npy</c>), read by <see cref="WorldCameraFile.Load"/>.</summary>
    public string? CameraPath { get; init; }

    /// <summary>A camera path in the model's action language, for models that declare <see cref="WorldCameraMode.ActionScript"/>.</summary>
    public string? CameraScript { get; init; }

    /// <summary>Intrinsics as <c>fx,fy,cx,cy</c> or the path of an intrinsics file (see <see cref="WorldCameraFile.LoadIntrinsics"/>);
    /// overrides the intrinsics in the camera file and needs a camera path or script.</summary>
    public string? Intrinsics { get; init; }

    /// <summary>Runs guidance as two sequential passes per step; refused by a model that does not declare the capability.</summary>
    public bool SequentialCfg { get; init; }

    /// <summary>Runs the model's refiner stage; refused by a model that does not declare one.</summary>
    public bool Refiner { get; init; }

    /// <summary>Asks for chunk-by-chunk streaming (true) or a whole-clip batch run (false); null takes the model's own mode.</summary>
    public bool? Streaming { get; init; }

    /// <summary>Runs a streaming session until it is stopped rather than for a frame count.</summary>
    public bool Unbounded { get; init; }

    /// <summary>Builds the request for <paramref name="initImage"/>, reading the camera and intrinsics files when they are named.</summary>
    /// <exception cref="ArgumentException">Intrinsics were given without a camera path or script, both a camera file and a script
    /// were given, or the intrinsics do not parse.</exception>
    public WorldRequest ToRequest(ImageData initImage, VramOverrides? vram = null)
    {
        ArgumentNullException.ThrowIfNull(initImage);
        if (CameraPath is not null && CameraScript is not null)
        {
            throw new ArgumentException("Give a camera path file or a camera action script, not both.", nameof(CameraScript));
        }
        WorldCameraPath? camera = CameraPath is not null ? WorldCameraFile.Load(CameraPath)
            : CameraScript is not null ? new WorldCameraPath { Script = CameraScript }
            : null;
        if (Intrinsics is not null)
        {
            // A live stream has no path up front: its intrinsics ride on an empty camera path that the actions extend.
            if (camera is null && (Streaming == true || Unbounded))
            {
                camera = new WorldCameraPath();
            }
            if (camera is null)
            {
                throw new ArgumentException("Intrinsics need a camera path or action script to apply to.", nameof(Intrinsics));
            }
            IReadOnlyList<WorldIntrinsics> intrinsics = WorldCameraFile.ParseIntrinsicsOption(Intrinsics);
            camera = camera with
            {
                Intrinsics = intrinsics.Count == 1 ? intrinsics[0] : null,
                FrameIntrinsics = intrinsics.Count == 1 ? null : intrinsics,
            };
        }
        return new WorldRequest
        {
            Prompt = Prompt ?? "",
            InitImage = initImage,
            Frames = Frames,
            Fps = Fps,
            Steps = Steps ?? 0,
            Guidance = Guidance,
            FlowShift = FlowShift,
            Seed = Seed,
            Camera = camera,
            SequentialCfg = SequentialCfg,
            Refiner = Refiner,
            Streaming = Streaming,
            Unbounded = Unbounded,
            Vram = vram,
        };
    }
}
