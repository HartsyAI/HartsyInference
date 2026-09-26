namespace HartsyInference.Diffusion.Sampling;

/// <summary>Resolves a sampler name to an <see cref="ISampler"/> over a given sigma array, and owns the vocabulary of
/// what is actually available.
///
/// <para><b>An unknown name throws.</b> The engine used to map any unrecognized sampler onto Euler with a log line —
/// so a workflow asking for <c>dpmpp_2m_sde_karras</c> silently got a different picture, and the user concluded the
/// engine was broken rather than that the sampler was missing. A named refusal is the same trade
/// <c>RecipeLoraMerge</c> already makes for a zero-match LoRA, for the same reason: the failure has to reach whoever can
/// act on it.</para></summary>
public static class SamplerRegistry
{
    /// <summary>Every sampler name <see cref="Create"/> accepts, in the ComfyUI vocabulary.</summary>
    public static IReadOnlyList<string> Names { get; } =
    [
        "euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive",
        "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_3m_sde", "ddpm", "ipndm", "ipndm_v",
        "deis", "res_multistep", "gradient_estimation", "er_sde", "seeds_2", "seeds_3", "sa_solver", "euler_cfg_pp",
        "uni_pc", "uni_pc_bh2",
    ];

    /// <summary>Samplers ComfyUI exposes that this engine does not implement yet, kept as data so the refusal message
    /// can distinguish "misspelled" from "known, not built".</summary>
    public static IReadOnlyList<string> NotYetImplemented { get; } =
    [
        "euler_ancestral_cfg_pp", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde_heun",
        "dpmpp_2m_sde_heun_gpu", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
        "gradient_estimation_cfg_pp", "exp_heun_2_x0", "exp_heun_2_x0_sde", "sa_solver_pece", "cfgpp_ud10_ab", "lcm",
    ];

    /// <summary>Samplers for which ComfyUI builds <c>steps + 1</c> sigmas and drops the penultimate one.</summary>
    public static IReadOnlyList<string> DiscardPenultimateSigma { get; } = ["dpm_2", "dpm_2_ancestral", "uni_pc", "uni_pc_bh2"];

    /// <summary>Whether <paramref name="name"/> resolves. Null/empty counts as available (it means the default).</summary>
    public static bool IsKnown(string? name)
    {
        string key = Canonical(name);
        return key.Length == 0 || Names.Contains(key, StringComparer.Ordinal) || Aliases.ContainsKey(key);
    }

    /// <summary>Short spellings people type, and ComfyUI's <c>_gpu</c> variants (the same solver with its Brownian
    /// tree on the GPU), mapped onto the canonical names.</summary>
    private static readonly Dictionary<string, string> Aliases = new(StringComparer.Ordinal)
    {
        ["euler_a"] = "euler_ancestral",
        ["dpmpp2m"] = "dpmpp_2m",
        ["dpm++2m"] = "dpmpp_2m",
        ["dpmpp2m_sde"] = "dpmpp_2m_sde",
        ["dpm++2m_sde"] = "dpmpp_2m_sde",
        ["dpmpp2s_a"] = "dpmpp_2s_ancestral",
        ["dpmpp_sde_gpu"] = "dpmpp_sde",
        ["dpmpp_2m_sde_gpu"] = "dpmpp_2m_sde",
        ["dpmpp_3m_sde_gpu"] = "dpmpp_3m_sde",
    };

    /// <summary>Canonical sampler name for <paramref name="name"/>, resolving aliases; empty for the default.</summary>
    public static string Resolve(string? name)
    {
        string key = Canonical(name);
        return Aliases.TryGetValue(key, out string? canonical) ? canonical : key;
    }

    /// <summary>Builds the named sampler over <paramref name="sigmas"/>. Null/empty resolves to <c>euler</c>.</summary>
    /// <param name="seed">Base seed for stochastic samplers; ignored by deterministic ones.</param>
    /// <param name="options">Family facts and test seams; null for defaults.</param>
    public static ISampler Create(string? name, float[] sigmas, int seed, SamplerOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(sigmas);
        string key = Resolve(name);
        return key switch
        {
            "" or "euler" => new EulerSampler(sigmas),
            "euler_ancestral" => new EulerAncestralSampler(sigmas, seed, 1.0f, options),
            "heun" => new HeunSampler(sigmas),
            "heunpp2" => new HeunPlusPlus2Sampler(sigmas, options),
            "dpm_2" => new Dpm2Sampler(sigmas, seed, 0f, options),
            "dpm_2_ancestral" => new Dpm2Sampler(sigmas, seed, 1.0f, options),
            "lms" => new LmsSampler(sigmas, options),
            "dpm_fast" => new DpmFastSampler(sigmas, options),
            "dpm_adaptive" => new DpmAdaptiveSampler(sigmas, options),
            "dpmpp_2s_ancestral" => new DpmPlusPlus2SAncestralSampler(sigmas, seed, 1.0f, options),
            "dpmpp_sde" => new DpmPlusPlusSdeSampler(sigmas, seed, 1.0f, options),
            "dpmpp_2m" => new DpmPlusPlus2MSampler(sigmas),
            "dpmpp_2m_sde" => new DpmPlusPlus2MSdeSampler(sigmas, seed, 1.0f, options),
            "dpmpp_3m_sde" => new DpmPlusPlus3MSdeSampler(sigmas, seed, 1.0f, options),
            "ddpm" => new DdpmSampler(sigmas, seed, options),
            "ipndm" => new IpndmSampler(sigmas, false, options),
            "ipndm_v" => new IpndmSampler(sigmas, true, options),
            "deis" => new DeisSampler(sigmas, options),
            "res_multistep" => new ResMultistepSampler(sigmas, options),
            "gradient_estimation" => new GradientEstimationSampler(sigmas, options),
            "er_sde" => new ErSdeSampler(sigmas, seed, options),
            "seeds_2" => new SeedsSampler(sigmas, seed, 2, 1.0f, options),
            "seeds_3" => new SeedsSampler(sigmas, seed, 3, 1.0f, options),
            "sa_solver" => new SaSolverSampler(sigmas, seed, options),
            "euler_cfg_pp" => new EulerCfgPlusPlusSampler(sigmas, options),
            "uni_pc" => new UniPcSampler(sigmas, false, options),
            "uni_pc_bh2" => new UniPcSampler(sigmas, true, options),
            _ => throw new NotSupportedException(Refusal(name, key)),
        };
    }

    /// <summary>The sigmas <paramref name="sampler"/> integrates: <paramref name="schedule"/> applied to the family's
    /// <paramref name="baseSigmas"/>, with ComfyUI's discard-penultimate rule for the samplers that use it.</summary>
    /// <remarks>The rule needs a <c>steps + 1</c> family grid: <paramref name="familyGrid"/> rebuilds it the way ComfyUI
    /// does when it starts at the same sigma, and otherwise it is interpolated piecewise-linearly in step fraction from
    /// the family's own grid, keeping both endpoints. It is skipped when the latent was already noised at the family's
    /// <c>sigma[startStep]</c> (img2img), since moving interior sigmas would desynchronise the two.</remarks>
    public static float[] BuildSigmas(string? sampler, string? schedule, float[] baseSigmas, bool startsFromNoisedInit = false,
        Func<int, float[]>? familyGrid = null)
    {
        ArgumentNullException.ThrowIfNull(baseSigmas);
        int steps = baseSigmas.Length - 1;
        bool discard = DiscardPenultimateSigma.Contains(Resolve(sampler), StringComparer.Ordinal) && !startsFromNoisedInit
            && steps >= 1 && baseSigmas[^1] == 0f;
        if (!discard)
        {
            return SigmaSchedule.Apply(schedule, baseSigmas);
        }
        float[]? rebuilt = familyGrid?.Invoke(steps + 1);
        // The latent is noised at baseSigmas[0]; a rebuilt grid starting elsewhere (diffusers "leading" spacing moves
        // the first timestep with the step count) would leave the sampler out of step with it.
        float[] dense = rebuilt?.Length == steps + 2 && rebuilt[0] == baseSigmas[0] ? rebuilt : Interpolated(baseSigmas, steps);
        float[] applied = SigmaSchedule.Apply(schedule, dense);
        float[] result = new float[steps + 1];
        Array.Copy(applied, result, steps);
        result[steps] = 0f;
        return result;
    }

    /// <summary>A <c>steps + 1</c> grid interpolated from a <c>steps</c> one, keeping both endpoints.</summary>
    private static float[] Interpolated(float[] baseSigmas, int steps)
    {
        float[] dense = new float[steps + 2];
        for (int j = 0; j <= steps + 1; j++)
        {
            double u = (double)j * steps / (steps + 1);
            int lo = Math.Min((int)u, steps - 1);
            double frac = u - lo;
            dense[j] = (float)((baseSigmas[lo] * (1.0 - frac)) + (baseSigmas[lo + 1] * frac));
        }
        dense[0] = baseSigmas[0];
        dense[^1] = 0f;
        return dense;
    }

    /// <summary>Builds the refusal message, separating a name this engine has simply not built from one it has never
    /// heard of — the first is a roadmap answer, the second is a typo.</summary>
    private static string Refusal(string? original, string key) =>
        NotYetImplemented.Contains(key, StringComparer.Ordinal)
            ? $"Sampler '{original}' is a recognized ComfyUI sampler that this engine has not implemented yet. "
                + $"Available now: {string.Join(", ", Names)}. Sigma schedules: {string.Join(", ", SigmaSchedule.Names)}."
            : $"Unknown sampler '{original}'. Available: {string.Join(", ", Names)}. "
                + $"Sigma schedules: {string.Join(", ", SigmaSchedule.Names)}.";

    /// <summary>Splits a compound ComfyUI selection like <c>dpmpp_2m_sde_karras</c> into its sampler and schedule
    /// halves, so a value pasted out of a shared workflow resolves instead of being rejected whole. Returns the input
    /// unchanged as the sampler when no schedule suffix is present.
    ///
    /// <para>Longest schedule suffix wins, because the names overlap: <c>ddim_uniform</c> ends with <c>uniform</c> and
    /// <c>sgm_uniform</c> does too, so a shortest-match rule would split <c>x_ddim_uniform</c> in the wrong place.</para></summary>
    public static (string Sampler, string? Schedule) SplitCompound(string? name)
    {
        string key = Canonical(name);
        if (key.Length == 0)
        {
            return ("", null);
        }
        string? bestSchedule = null;
        int bestLength = 0;
        foreach (string schedule in SigmaSchedule.Names)
        {
            if (schedule == "normal")
            {
                continue;
            }
            string suffix = "_" + schedule;
            if (key.EndsWith(suffix, StringComparison.Ordinal) && key.Length > suffix.Length && suffix.Length > bestLength)
            {
                bestSchedule = schedule;
                bestLength = suffix.Length;
            }
        }
        return bestSchedule is null ? (key, null) : (key[..^bestLength], bestSchedule);
    }

    private static string Canonical(string? name) => (name ?? "").Trim().ToLowerInvariant();
}
