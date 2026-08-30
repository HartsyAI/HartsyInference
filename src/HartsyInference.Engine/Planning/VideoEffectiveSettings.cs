using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Planning;

/// <summary>The fully resolved sampling and geometry values used by construction and execution.</summary>
public sealed record VideoEffectiveSettings
{
    /// <summary>Aligned output width in pixels.</summary>
    public required int Width { get; init; }

    /// <summary>Aligned output height in pixels.</summary>
    public required int Height { get; init; }

    /// <summary>Aligned output frame count.</summary>
    public required int Frames { get; init; }

    /// <summary>Playback and model frame rate.</summary>
    public required int Fps { get; init; }

    /// <summary>Denoising evaluation count.</summary>
    public required int Steps { get; init; }

    /// <summary>Classifier-free guidance scale.</summary>
    public required float CfgScale { get; init; }

    /// <summary>Video flow shift, or null when a legacy family resolves it inside its own pipeline.</summary>
    public float? FlowShift { get; init; }

    /// <summary>Audio flow shift, or null for a family without a declared joint-AV shift.</summary>
    public float? AudioFlowShift { get; init; }

    /// <summary>Numerical sampler name, or null when the family owns solver selection internally.</summary>
    public string? Sampler { get; init; }

    /// <summary>Sigma scheduler name, or null when the family owns schedule selection internally.</summary>
    public string? Scheduler { get; init; }

    /// <summary>Concrete seed; planning replaces a negative request seed before execution.</summary>
    public required long Seed { get; init; }

    /// <summary>Reference-media sizing selected by the profile or request.</summary>
    public required VideoReferenceSizing ReferenceSizing { get; init; }

    /// <summary>Sampling fields the profile fixes to these values.</summary>
    public required VideoLockedFields LockedFields { get; init; }

    /// <summary>Applies the plan values while retaining all media and composition inputs from the request.</summary>
    public VideoRequest Apply(VideoRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        return request with
        {
            Width = Width,
            Height = Height,
            Frames = Frames,
            Fps = Fps,
            Steps = Steps,
            CfgScale = CfgScale,
            FlowShift = FlowShift,
            AudioFlowShift = AudioFlowShift,
            Sampler = Sampler,
            Scheduler = Scheduler,
            Seed = Seed,
            ReferenceSizing = ReferenceSizing,
        };
    }
}
