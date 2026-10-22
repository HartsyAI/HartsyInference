using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes;

/// <summary>One music family's recommended generation settings, resolving the <see cref="MusicRequest"/> fields the
/// caller left <c>null</c>. A <c>null</c> member means the family's loader or pipeline owns that choice.</summary>
public sealed record MusicDefaults
{
    /// <summary>Target clip length in seconds; every music family starts from the same 10 s clip.</summary>
    public required double Duration { get; init; }

    /// <summary>Diffusion or flow steps; null where the family has no step count or its pipeline picks one.</summary>
    public int? Steps { get; init; }

    /// <summary>Noise-schedule shift (ACE-Step); null where the family has none.</summary>
    public double? Shift { get; init; }

    /// <summary>Classifier-free guidance scale; null where the pipeline applies its own guidance.</summary>
    public double? CfgScale { get; init; }

    /// <summary>Start of the CFG active interval over the schedule (ACE-Step base/sft).</summary>
    public double? CfgIntervalStart { get; init; }

    /// <summary>End of the CFG active interval over the schedule (ACE-Step base/sft).</summary>
    public double? CfgIntervalEnd { get; init; }

    /// <summary>Cover-mode strength (ACE-Step); null where cover editing does not exist.</summary>
    public double? CoverStrength { get; init; }

    /// <summary>Start second of the reference-audio span; null where the family takes no reference.</summary>
    public double? ReferenceStartSeconds { get; init; }

    /// <summary>End second of the reference-audio span; null where the family takes no reference.</summary>
    public double? ReferenceEndSeconds { get; init; }

    /// <summary>ODE or SDE solver (ACE-Step); null where the family has no solver choice.</summary>
    public string? InferMethod { get; init; }

    /// <summary>Guidance blend, <c>apg</c>, <c>cfg</c> or <c>adg</c> (ACE-Step); null where the family has none.</summary>
    public string? GuidanceType { get; init; }

    /// <summary>LM planner chain-of-thought phase (ACE-Step 5 Hz planner).</summary>
    public bool? Thinking { get; init; }

    /// <summary>LM planner sampling temperature (ACE-Step 5 Hz planner).</summary>
    public double? LmTemperature { get; init; }

    /// <summary>LM planner CFG scale (ACE-Step 5 Hz planner).</summary>
    public double? LmCfgScale { get; init; }

    /// <summary>LM planner top-K cutoff (ACE-Step 5 Hz planner).</summary>
    public int? LmTopK { get; init; }

    /// <summary>LM planner nucleus threshold (ACE-Step 5 Hz planner).</summary>
    public double? LmTopP { get; init; }

    /// <summary>Revision of these values; bump it when a default changes so a stored plan can tell the two apart.</summary>
    public int Revision { get; init; } = 1;

    /// <summary>Returns <paramref name="request"/> with every unset field filled from these defaults; values the caller
    /// supplied are left untouched.</summary>
    public MusicRequest Apply(MusicRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        return request with
        {
            Duration = request.Duration ?? Duration,
            InferSteps = request.InferSteps ?? Steps,
            Shift = request.Shift ?? Shift,
            CfgScale = request.CfgScale ?? CfgScale,
            CfgIntervalStart = request.CfgIntervalStart ?? CfgIntervalStart,
            CfgIntervalEnd = request.CfgIntervalEnd ?? CfgIntervalEnd,
            CoverStrength = request.CoverStrength ?? CoverStrength,
            ReferenceStartSeconds = request.ReferenceStartSeconds ?? ReferenceStartSeconds,
            ReferenceEndSeconds = request.ReferenceEndSeconds ?? ReferenceEndSeconds,
            InferMethod = request.InferMethod ?? InferMethod,
            GuidanceType = request.GuidanceType ?? GuidanceType,
            Thinking = request.Thinking ?? Thinking,
            LmTemperature = request.LmTemperature ?? LmTemperature,
            LmCfgScale = request.LmCfgScale ?? LmCfgScale,
            LmTopK = request.LmTopK ?? LmTopK,
            LmTopP = request.LmTopP ?? LmTopP,
        };
    }
}
