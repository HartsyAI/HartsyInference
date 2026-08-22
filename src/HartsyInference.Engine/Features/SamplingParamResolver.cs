using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Features;

/// <summary>Centralized resolution of the request's sampling knobs into the concrete numbers a pipeline takes: effective step count (honoring <see cref="ImageRequest.EndStepsEarly"/>), scheduler name, and CLIP-skip.</summary>
public static class SamplingParamResolver
{
    /// <summary>Effective step count for a single-stage generation, always at least 1. <c>EndStepsEarly</c> is the fraction of the configured steps to cut off (Steps=20, EndStepsEarly=0.25 → 15 steps); the truncating cast matches ComfyUI's <c>endStep = (int)(steps * (1 - endEarly))</c> so the two backends agree.</summary>
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

    /// <summary>Resolves the request's sampler/scheduler choice into the string a pipeline consumes — a <c>SchedulerFactory</c> name, a <c>Sampling.SamplerRegistry</c> name, or a compound <c>sampler_schedule</c> selection, passed through for the pipeline to split. <para><b>An unrecognized name now throws.</b> This method used to map anything it did not know onto Euler and write a <c>Logs.Verbose</c> line, on the reasoning that "sampler choice is a preference, not a correctness contract". For someone migrating a ComfyUI workflow it IS a correctness contract: they ask for <c>dpmpp_2m_sde_karras</c>, silently receive a Euler image, and conclude the engine is broken rather than that the sampler is missing. The refusal names the value and lists what exists, matching the trade <c>LoraApplier</c> already makes for a zero-match LoRA.</para></summary>
    /// <exception cref="NotSupportedException">The requested sampler or sigma schedule is not available.</exception>
    public static string? ResolveSchedulerName(ImageRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        string? requested = !string.IsNullOrWhiteSpace(request.Sampler) ? request.Sampler
            : !string.IsNullOrWhiteSpace(request.Scheduler) ? request.Scheduler
            : null;
        if (requested is null)
        {
            return null;
        }

        (string samplerName, string? scheduleName) = SamplerRegistry.SplitCompound(requested);
        if (!SamplerRegistry.IsKnown(samplerName) && !SchedulerFactory.IsKnown(samplerName)
            && MapSamplerName(samplerName) is null && !IsLegacyEuler(samplerName))
        {
            throw new NotSupportedException(
                $"Sampler '{requested}' is not available. Samplers: "
                + $"{string.Join(", ", SamplerRegistry.Names.Concat(SchedulerFactory.Names).Distinct(StringComparer.Ordinal))}. "
                + $"Sigma schedules: {string.Join(", ", SigmaSchedule.Names)}.");
        }
        if (!SigmaSchedule.IsKnown(scheduleName))
        {
            throw new NotSupportedException(
                $"Sigma schedule '{scheduleName}' is not available. Schedules: {string.Join(", ", SigmaSchedule.Names)}.");
        }

        // Pass the selection through as given. Pipelines carrying the sampler seam split it themselves via
        // SamplerRegistry.SplitCompound; the legacy SD-family path maps the sampler half onto a SchedulerFactory name.
        return SamplerRegistry.IsKnown(samplerName) || scheduleName is not null
            ? requested.Trim().ToLowerInvariant()
            : MapSamplerName(samplerName);
    }

    /// <summary><c>euler</c> maps to null (the factory default), which is indistinguishable from "unmapped" in <see cref="MapSamplerName"/>'s return — so the validation above has to ask about it separately.</summary>
    private static bool IsLegacyEuler(string name) => string.Equals(name, "euler", StringComparison.OrdinalIgnoreCase);

    /// <summary>Converts the request's CLIP-skip into the layers-from-end convention (1 = final layer, 2 = penultimate). Hosts using the negative-from-end convention (-1 = final) are accepted too. Returns null when unset or default. Only SD 1.5 honors this — SDXL is penultimate by spec, matching ComfyUI.</summary>
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
