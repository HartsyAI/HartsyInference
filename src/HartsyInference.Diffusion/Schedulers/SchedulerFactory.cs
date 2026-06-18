using HartsyInference.Core.Schedulers;

namespace HartsyInference.Diffusion.Schedulers;

/// <summary>Creates the requested noise scheduler by name. The legacy short-form names accepted here (<c>"ddim"</c>, <c>"dpm++2m"</c> / <c>"dpmpp2m"</c>, <c>"euler"</c>, <c>"lcm"</c>) match what the SD1.5 / SDXL pipelines historically accepted; new pipelines should route their <c>request.Scheduler</c> string through this helper instead of duplicating the switch.
/// <para>Flow-matching transformers (Flux, SD3, Lumina2 family) use <see cref="FlowMatchEulerDiscreteScheduler"/> directly with dynamic-shift configuration; they don't go through this factory because their scheduler choice isn't user-selectable.</para>
/// </summary>
public static class SchedulerFactory
{
    /// <summary>Returns the requested epsilon/v-prediction scheduler for SD1.5- / SDXL-style sigma-domain sampling. Defaults to <see cref="EulerDiscreteScheduler"/> when <paramref name="name"/> is null, empty, or unrecognized — the same fallback the legacy pipelines used.</summary>
    public static IScheduler Create(string? name) => (name?.ToLowerInvariant()) switch
    {
        "ddim" => new DdimScheduler(),
        "dpm++2m" or "dpmpp2m" => new DpmPlusPlus2MScheduler(),
        "lcm" => new LcmScheduler(),
        "tcd" => new TcdScheduler(),
        _ => new EulerDiscreteScheduler(),
    };
}
