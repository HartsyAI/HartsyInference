using HartsyInference.Core.Memory.Vram;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Worlds;

/// <summary>The flat option set a world command passes in (CLI flags, or the same fields from any other transport), mapped
/// to a <see cref="WorldRequest"/>. A null field means "the model default", so nothing is filled in on the caller's behalf.</summary>
public sealed record WorldRequestOptions
{
    /// <summary>Frames to roll out, counting the seed frame.</summary>
    public int? Frames { get; init; }

    /// <summary>Output frame rate.</summary>
    public int? Fps { get; init; }

    /// <summary>Denoising steps per frame.</summary>
    public int? Steps { get; init; }

    /// <summary>RNG seed; negative draws one.</summary>
    public long Seed { get; init; } = -1;

    /// <summary>Path to a camera path file (<c>.json</c> or <c>.npy</c>), read by <see cref="WorldCameraFile.Load"/>.</summary>
    public string? CameraPath { get; init; }

    /// <summary>Intrinsics as <c>fx,fy,cx,cy</c>; overrides the intrinsics in the camera file and needs a camera path.</summary>
    public string? Intrinsics { get; init; }

    /// <summary>Builds the request for <paramref name="initImage"/>, reading the camera file when one is named.</summary>
    /// <exception cref="ArgumentException">Intrinsics were given without a camera path, or do not parse.</exception>
    public WorldRequest ToRequest(ImageData initImage, VramOverrides? vram = null)
    {
        ArgumentNullException.ThrowIfNull(initImage);
        WorldCameraPath? camera = CameraPath is null ? null : WorldCameraFile.Load(CameraPath);
        if (Intrinsics is not null)
        {
            if (camera is null)
            {
                throw new ArgumentException("Intrinsics need a camera path to apply to.", nameof(Intrinsics));
            }
            camera = new WorldCameraPath { Poses = camera.Poses, Intrinsics = WorldCameraFile.ParseIntrinsics(Intrinsics) };
        }
        return new WorldRequest
        {
            InitImage = initImage,
            Frames = Frames,
            Fps = Fps,
            Steps = Steps ?? 0,
            Seed = Seed,
            Camera = camera,
            Vram = vram,
        };
    }
}
