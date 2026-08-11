using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Features;

/// <summary>Centralized resolution of the request's sampling knobs into the concrete numbers a pipeline takes: effective
/// step count (honoring <see cref="ImageRequest.EndStepsEarly"/>), scheduler name, and CLIP-skip.</summary>
public static class SamplingParamResolver
{
    /// <summary>Effective step count for a single-stage generation, always at least 1. <c>EndStepsEarly</c> is the
    /// fraction of the configured steps to cut off (Steps=20, EndStepsEarly=0.25 → 15 steps); the truncating cast matches
    /// ComfyUI's <c>endStep = (int)(steps * (1 - endEarly))</c> so the two backends agree.</summary>
    public static int ResolveSteps(ImageRequest request, int fallback)
    {
        ArgumentNullException.ThrowIfNull(request);
        int steps = request.Steps ?? fallback;
        double endEarly = request.EndStepsEarly ?? 0.0;
        if (endEarly > 0)
        {
            int reduced = (int)(steps * (1 - endEarly));
            if (reduced < 1)
            {
                Logs.Warning(
                    $"[Features][Sampling] EndStepsEarly={endEarly} would zero out the step count (steps={steps}); "
                    + "clamping to 1 step. Consider lowering EndStepsEarly.");
                reduced = 1;
            }
            steps = reduced;
        }
        return steps;
    }

    /// <summary>Maps the request's sampler/scheduler choice onto a <c>SchedulerFactory</c> name (<c>"ddim"</c> /
    /// <c>"dpm++2m"</c> / <c>"lcm"</c>), or null for the default Euler. Only meaningful for the sigma-domain SD-family
    /// pipelines; flow-match architectures use their canonical scheduler and ignore it. Unmappable names (euler_ancestral,
    /// SDE variants, uni_pc, …) fall back to Euler with a log rather than a refusal — sampler choice is a preference,
    /// not a correctness contract.</summary>
    public static string? ResolveSchedulerName(ImageRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (!string.IsNullOrWhiteSpace(request.Sampler))
        {
            return MapSamplerName(request.Sampler);
        }
        if (!string.IsNullOrWhiteSpace(request.Scheduler))
        {
            return MapSamplerName(request.Scheduler);
        }
        return null;
    }

    /// <summary>Converts the request's CLIP-skip into the layers-from-end convention (1 = final layer, 2 = penultimate).
    /// Hosts using the negative-from-end convention (-1 = final) are accepted too. Returns null when unset or default.
    /// Only SD 1.5 honors this — SDXL is penultimate by spec, matching ComfyUI.</summary>
    public static int? ResolveClipSkip(ImageRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        int stopAt = request.ClipSkip ?? 0;
        if (stopAt is 0 or -1)
        {
            return null;
        }
        return stopAt > 0 ? Math.Clamp(stopAt, 1, 12) : Math.Clamp(-stopAt, 1, 12);
    }

    private static string? MapSamplerName(string name) => name.ToLowerInvariant() switch
    {
        "euler" => null, // SchedulerFactory default
        "ddim" => "ddim",
        "dpm++2m" or "dpmpp_2m" or "dpmpp2m" => "dpm++2m",
        "lcm" => "lcm",
        "tcd" => "tcd",
        _ => LogUnmapped(name),
    };

    private static string? LogUnmapped(string name)
    {
        Logs.Verbose($"[Features][Sampling] Sampler '{name}' isn't available (have: euler, ddim, dpm++2m, lcm, tcd) — using Euler.");
        return null;
    }
}
