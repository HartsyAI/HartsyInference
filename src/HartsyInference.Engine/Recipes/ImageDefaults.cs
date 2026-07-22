using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes;

/// <summary>One image family's creator-recommended sampling settings — the values a <see cref="ImageRequest"/> tunable
/// left <c>null</c> resolves to, so a distilled/turbo architecture runs at its own official numbers instead of a
/// generic 20-step / CFG-7.5 guess while any value the caller does set still wins.</summary>
public sealed record ImageDefaults
{
    /// <summary>Denoising steps the architecture's authors sample with.</summary>
    public required int Steps { get; init; }

    /// <summary>Guidance scale the architecture's authors sample with; 1.0 means guidance-free (distilled) sampling.</summary>
    public required float CfgScale { get; init; }

    /// <summary>Sampler name the family expects; null leaves the pipeline's canonical sampler in place.</summary>
    public string? Sampler { get; init; }

    /// <summary>Scheduler / sigma-schedule name the family expects; null leaves the pipeline's canonical schedule in place.</summary>
    public string? Scheduler { get; init; }

    /// <summary>Flow-match sigma shift the family trains with; null leaves the pipeline's built-in shift in place.</summary>
    public double? SigmaShift { get; init; }

    /// <summary>Native training width in pixels; null falls back to <see cref="Standard"/>'s 1024.</summary>
    public int? Width { get; init; }

    /// <summary>Native training height in pixels; null falls back to <see cref="Standard"/>'s 1024.</summary>
    public int? Height { get; init; }

    /// <summary>The generic fallback (20 steps, CFG 7.5, 1024²) a recipe that has not declared its own numbers inherits.</summary>
    public static ImageDefaults Standard { get; } = new ImageDefaults { Steps = 20, CfgScale = 7.5f, Width = 1024, Height = 1024 };

    /// <summary>Returns <paramref name="request"/> with every unset tunable filled in from these defaults; values the caller supplied are left untouched.</summary>
    public ImageRequest Apply(ImageRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        return request with
        {
            Steps = request.Steps ?? Steps,
            CfgScale = request.CfgScale ?? CfgScale,
            Sampler = request.Sampler ?? Sampler,
            Scheduler = request.Scheduler ?? Scheduler,
            SigmaShift = request.SigmaShift ?? SigmaShift,
            Width = request.Width ?? Width ?? Standard.Width,
            Height = request.Height ?? Height ?? Standard.Height,
        };
    }
}
