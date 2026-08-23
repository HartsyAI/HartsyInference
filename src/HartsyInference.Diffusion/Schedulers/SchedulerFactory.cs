using HartsyInference.Core.Schedulers;

namespace HartsyInference.Diffusion.Schedulers;

/// <summary>Creates the requested noise scheduler by name. The legacy short-form names accepted here (<c>"ddim"</c>, <c>"dpm++2m"</c> / <c>"dpmpp2m"</c>, <c>"euler"</c>, <c>"lcm"</c>) match what the SD1.5 / SDXL pipelines historically accepted; new pipelines should route their <c>request.Scheduler</c> string through this helper instead of duplicating the switch.
/// <para>Flow-matching transformers (Flux, SD3, Lumina2 family) use <see cref="FlowMatchEulerDiscreteScheduler"/> directly with dynamic-shift configuration; they don't go through this factory because their scheduler choice isn't user-selectable.</para></summary>
public static class SchedulerFactory
{
    /// <summary>Every name <see cref="Create"/> resolves to something other than the Euler default. Callers validate
    /// against this (together with <c>Sampling.SamplerRegistry.Names</c>) so an unrecognized sampler can be refused by
    /// name instead of silently becoming Euler.</summary>
    public static IReadOnlyList<string> Names { get; } = ["euler", "ddim", "dpm++2m", "dpmpp2m", "lcm", "tcd"];

    /// <summary>Whether <paramref name="name"/> is one this factory genuinely implements — null/empty counts, since
    /// that means "the default".</summary>
    public static bool IsKnown(string? name)
    {
        string key = (name ?? "").Trim().ToLowerInvariant();
        return key.Length == 0 || Names.Contains(key, StringComparer.Ordinal);
    }

    /// <summary>Returns the requested epsilon/v-prediction scheduler for SD1.5- / SDXL-style sigma-domain sampling.
    /// Null or empty gives <see cref="EulerDiscreteScheduler"/>.
    ///
    /// <para><b>An unrecognized name throws.</b> This used to return Euler for anything it did not know, which is how a
    /// request for <c>euler_ancestral</c> on a family without the sampler seam turned into a silently different image.
    /// Callers that legitimately want the Euler base for a <c>Sampling.SamplerRegistry</c> sampler — the sampler
    /// supplies the integrator, this supplies the sigma range — must pass null explicitly rather than relying on a
    /// fallback that cannot distinguish that case from a typo.</para></summary>
    /// <exception cref="NotSupportedException"><paramref name="name"/> is not a scheduler this factory implements.</exception>
    public static IScheduler Create(string? name) => (name?.Trim().ToLowerInvariant()) switch
    {
        null or "" or "euler" => new EulerDiscreteScheduler(),
        "ddim" => new DdimScheduler(),
        "dpm++2m" or "dpmpp2m" or "dpmpp_2m" => new DpmPlusPlus2MScheduler(),
        "lcm" => new LcmScheduler(),
        "tcd" => new TcdScheduler(),
        _ => throw new NotSupportedException(
            $"Sampler '{name}' is not available on this model family. This family runs the legacy scheduler path, "
            + $"which implements: {string.Join(", ", Names)}. The broader sampler set "
            + "(euler_ancestral, heun, dpm_2, lms, dpmpp_2s_ancestral, dpmpp_2m, dpmpp_2m_sde) and the sigma "
            + "schedules are available on families carrying the sampler seam — SDXL today, the rest as they are "
            + "converted."),
    };
}
