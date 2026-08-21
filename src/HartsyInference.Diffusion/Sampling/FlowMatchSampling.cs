using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Schedulers;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>Resolves a request's sampler selection for a flow-matching family — the DiT fleet (Flux, Flux.2,
/// Qwen-Image, Krea2, Z-Image, Chroma, HiDream, Lumina2, Ideogram 4, Mage-Flow, …), which is 20+ of the engine's 26
/// image families and had NO user-selectable sampler at all before this.
///
/// <para>The family keeps owning its sigma range: its own dynamic shift, <c>shift_terminal</c>, or terminal stretch
/// produced the array this re-spaces. A schedule name changes the spacing between the family's endpoints and never
/// replaces them, which is why <c>karras</c> on Flux stays a Flux schedule rather than becoming an SDXL one.</para></summary>
public static class FlowMatchSampling
{
    /// <summary>Whether <paramref name="selection"/> asks for anything other than the family's default Euler. Pipelines
    /// use this to narrow incompatible fast paths (step-graph capture, step-cache, DiT sharding) for that generation
    /// only, rather than refusing the sampler outright.</summary>
    public static bool IsNonDefault(string? selection)
    {
        (string sampler, string? schedule) = SamplerRegistry.SplitCompound(selection);
        return schedule is not null || (sampler.Length > 0 && !string.Equals(sampler, "euler", StringComparison.Ordinal));
    }

    /// <summary>Builds the sampler for <paramref name="selection"/> over <paramref name="scheduler"/>'s own sigmas.
    /// Null or <c>euler</c> gives <see cref="EulerSampler"/>, whose step is the same fused
    /// <see cref="Core.Backends.IBackend.CfgEulerStep"/> device op the pipeline called before — so a request that names
    /// no sampler is unchanged, which is the property the whole rollout rests on.</summary>
    /// <exception cref="NotSupportedException">The sampler or sigma schedule is not available.</exception>
    /// <param name="startsFromNoisedInit">True when the initial latent was already noised at the FAMILY's
    /// <c>sigma[startStep]</c> — img2img and inpaint. A re-spaced schedule is refused in that case: the latent would
    /// carry the family's noise level while the sampler integrates from the re-spaced one, and a mismatched starting
    /// sigma yields coherent-but-wrong output with nothing to point at. Fixing it properly means handing the re-spaced
    /// array to the init-latent builder, which is a larger change than this rollout.</param>
    public static ISampler Resolve(string? selection, FlowMatchEulerDiscreteScheduler scheduler, int seed, string family,
        bool startsFromNoisedInit = false)
    {
        ArgumentNullException.ThrowIfNull(scheduler);
        (string samplerName, string? scheduleName) = SamplerRegistry.SplitCompound(selection);
        if (startsFromNoisedInit && !string.IsNullOrEmpty(scheduleName))
        {
            throw new NotSupportedException(
                $"Sigma schedule '{scheduleName}' cannot be combined with img2img or inpaint on {family} yet: the init "
                + "latent is noised at the family's own sigma[startStep], so a re-spaced schedule would start the "
                + "sampler from a different noise level than the latent actually carries. Use the schedule on a "
                + "text-to-image generation, or drop the schedule suffix.");
        }
        if (!SamplerRegistry.IsKnown(samplerName))
        {
            throw new NotSupportedException(
                $"Sampler '{selection}' is not available on {family}. Samplers: {string.Join(", ", SamplerRegistry.Names)}. "
                + $"Sigma schedules: {string.Join(", ", SigmaSchedule.Names)}.");
        }
        if (!SigmaSchedule.IsKnown(scheduleName))
        {
            throw new NotSupportedException(
                $"Sigma schedule '{scheduleName}' is not available. Schedules: {string.Join(", ", SigmaSchedule.Names)}.");
        }
        float[] sigmas = SigmaSchedule.Apply(scheduleName, scheduler.Sigmas());
        ISampler sampler = SamplerRegistry.Create(samplerName, sigmas, seed);
        if (IsNonDefault(selection))
        {
            Logs.Info($"[Sampling] {family}: sampler={sampler.Name}, schedule={scheduleName ?? "normal"}.");
        }
        return sampler;
    }
}
