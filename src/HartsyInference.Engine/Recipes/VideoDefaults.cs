using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes;

/// <summary>One video family's creator-recommended sampling settings — the video counterpart of
/// <see cref="ImageDefaults"/>, resolving the <see cref="VideoRequest"/> tunables the caller left <c>null</c>.</summary>
public sealed record VideoDefaults
{
    /// <summary>Denoising steps the architecture's authors sample with.</summary>
    public required int Steps { get; init; }

    /// <summary>Guidance scale the architecture's authors sample with.</summary>
    public required float CfgScale { get; init; }

    /// <summary>Native training width in pixels; null falls back to <see cref="Standard"/>'s 704.</summary>
    public int? Width { get; init; }

    /// <summary>Native training height in pixels; null falls back to <see cref="Standard"/>'s 480.</summary>
    public int? Height { get; init; }

    /// <summary>Native/recommended frame count; null falls back to <see cref="Standard"/>'s 25.</summary>
    public int? Frames { get; init; }

    /// <summary>Native/recommended frame rate; null falls back to <see cref="Standard"/>'s 25.</summary>
    public int? Fps { get; init; }

    /// <summary>The generic fallback (30 steps, CFG 6.0, 704x480, 25 frames @ 25 fps) a recipe that has not declared
    /// its own numbers inherits.</summary>
    public static VideoDefaults Standard { get; } =
        new VideoDefaults { Steps = 30, CfgScale = 6.0f, Width = 704, Height = 480, Frames = 25, Fps = 25 };

    /// <summary>Returns <paramref name="request"/> with every unset tunable filled in from these defaults; values the caller supplied are left untouched.</summary>
    public VideoRequest Apply(VideoRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        return request with
        {
            Steps = request.Steps ?? Steps,
            CfgScale = request.CfgScale ?? CfgScale,
            Width = request.Width ?? Width ?? Standard.Width,
            Height = request.Height ?? Height ?? Standard.Height,
            Frames = request.Frames ?? Frames ?? Standard.Frames,
            Fps = request.Fps ?? Fps ?? Standard.Fps,
        };
    }
}
