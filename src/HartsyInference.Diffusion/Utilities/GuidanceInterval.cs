using HartsyInference.Core.Configuration;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Noise-level band in which classifier-free guidance applies (limited-interval guidance,
/// arXiv:2404.07724 — see docs/Research/STEP_ACCELERATION.md §3.4). Guidance is harmful at high noise and
/// unnecessary at low noise; outside <c>[Start, End]</c> the pipeline skips the unconditional forward
/// entirely and uses the conditional prediction alone — halving that step's compute with a measured
/// quality IMPROVEMENT in the reference paper. Same convention as ACE-Step 1.5's cfg-interval: applies when
/// <c>Start ≤ σ ≤ End</c> on a monotone normalized noise level σ ∈ [0, 1] (flow-matching t/1000, or
/// step-schedule sigma normalized to its own range).</summary>
public readonly record struct GuidanceInterval(float Start, float End)
{
    /// <summary>The full band — guidance applies at every step (the no-op default).</summary>
    public static GuidanceInterval Always => new(0f, 1f);

    /// <summary>True when the band covers everything, so per-step gating can be skipped.</summary>
    public bool IsAlways => Start <= 0f && End >= 1f;

    /// <summary>Whether guidance applies at this normalized noise level.</summary>
    public bool Applies(float sigma) => sigma >= Start && sigma <= End;

    /// <summary>Parses "start,end" (e.g. "0.15,0.9"). Throws on malformed input or an inverted/out-of-range band.</summary>
    public static GuidanceInterval Parse(string spec)
    {
        if (string.IsNullOrWhiteSpace(spec))
            throw new ArgumentException("GuidanceInterval spec must be 'start,end'.", nameof(spec));
        string[] parts = spec.Split(',', StringSplitOptions.TrimEntries);
        if (parts.Length != 2
            || !float.TryParse(parts[0], System.Globalization.CultureInfo.InvariantCulture, out float start)
            || !float.TryParse(parts[1], System.Globalization.CultureInfo.InvariantCulture, out float end))
            throw new ArgumentException($"GuidanceInterval spec '{spec}' must be 'start,end' with two floats.", nameof(spec));
        if (start < 0f || end > 1f || start >= end)
            throw new ArgumentException($"GuidanceInterval '{spec}' must satisfy 0 ≤ start < end ≤ 1.", nameof(spec));
        return new GuidanceInterval(start, end);
    }

    /// <summary>Reads the interval from <see cref="EngineKnobs.CfgInterval"/> ("start,end"), returning
    /// <see cref="Always"/> when unset/empty so the default path is byte-identical to pre-feature behavior. A
    /// malformed value throws — silently ignoring a mistyped perf knob would invalidate an A/B run.</summary>
    public static GuidanceInterval FromEnvironment()
    {
        string? spec = EngineKnobs.CfgInterval.Value;
        return string.IsNullOrWhiteSpace(spec) ? Always : Parse(spec);
    }
}
