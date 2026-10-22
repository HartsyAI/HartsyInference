using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes;

/// <summary>One image family's creator-recommended sampling settings — the values a <see cref="ImageRequest"/> tunable
/// left <c>null</c> resolves to; any value the caller does set wins.</summary>
public sealed record ImageDefaults
{
    /// <summary>Revision of these values, recorded on every plan and result that used them. Bump it whenever a value
    /// changes, so output generated before and after the change can be told apart.</summary>
    public int Revision { get; init; } = 1;

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

    /// <summary>Native training width in pixels.</summary>
    public required int Width { get; init; }

    /// <summary>Native training height in pixels.</summary>
    public required int Height { get; init; }

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
            Width = request.Width ?? Width,
            Height = request.Height ?? Height,
        };
    }
}
