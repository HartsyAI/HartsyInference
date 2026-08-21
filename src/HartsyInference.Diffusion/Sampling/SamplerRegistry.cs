namespace HartsyInference.Diffusion.Sampling;

/// <summary>Resolves a sampler name to an <see cref="ISampler"/> over a given sigma array, and owns the vocabulary of
/// what is actually available.
///
/// <para><b>An unknown name throws.</b> The engine used to map any unrecognized sampler onto Euler with a log line —
/// so a workflow asking for <c>dpmpp_2m_sde_karras</c> silently got a different picture, and the user concluded the
/// engine was broken rather than that the sampler was missing. A named refusal is the same trade
/// <c>LoraApplier</c> already makes for a zero-match LoRA, for the same reason: the failure has to reach whoever can
/// act on it.</para></summary>
public static class SamplerRegistry
{
    /// <summary>Every sampler name <see cref="Create"/> accepts, in the ComfyUI vocabulary.</summary>
    public static IReadOnlyList<string> Names { get; } =
    [
        "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", "lms",
        "dpmpp_2s_ancestral", "dpmpp_2m", "dpmpp_2m_sde",
    ];

    /// <summary>Samplers ComfyUI exposes that this engine does not implement yet, kept as data so the refusal message
    /// can distinguish "misspelled" from "known, not built" — a materially different thing for whoever hits it.</summary>
    public static IReadOnlyList<string> NotYetImplemented { get; } =
    [
        "dpmpp_sde", "dpmpp_3m_sde", "ddpm", "ipndm", "ipndm_v", "deis", "res_multistep",
        "gradient_estimation", "uni_pc", "uni_pc_bh2", "sa_solver", "er_sde", "seeds_2", "seeds_3",
        "heunpp2", "euler_cfg_pp", "dpm_fast", "dpm_adaptive",
    ];

    /// <summary>Whether <paramref name="name"/> resolves. Null/empty counts as available (it means the default).</summary>
    public static bool IsKnown(string? name)
    {
        string key = Canonical(name);
        return key.Length == 0 || Names.Contains(key, StringComparer.Ordinal) || Aliases.ContainsKey(key);
    }

    /// <summary>Short spellings people actually type, mapped onto the canonical names.</summary>
    private static readonly Dictionary<string, string> Aliases = new(StringComparer.Ordinal)
    {
        ["euler_a"] = "euler_ancestral",
        ["dpmpp2m"] = "dpmpp_2m",
        ["dpm++2m"] = "dpmpp_2m",
        ["dpmpp2m_sde"] = "dpmpp_2m_sde",
        ["dpm++2m_sde"] = "dpmpp_2m_sde",
        ["dpmpp2s_a"] = "dpmpp_2s_ancestral",
    };

    /// <summary>Builds the named sampler over <paramref name="sigmas"/>. Null/empty resolves to <c>euler</c>.</summary>
    /// <param name="seed">Base seed for stochastic samplers; ignored by deterministic ones.</param>
    public static ISampler Create(string? name, float[] sigmas, int seed)
    {
        ArgumentNullException.ThrowIfNull(sigmas);
        string key = Canonical(name);
        if (Aliases.TryGetValue(key, out string? canonical))
        {
            key = canonical;
        }
        return key switch
        {
            "" or "euler" => new EulerSampler(sigmas),
            "euler_ancestral" => new EulerAncestralSampler(sigmas, seed),
            "heun" => new HeunSampler(sigmas),
            "dpm_2" => new Dpm2Sampler(sigmas, seed, eta: 0f),
            "dpm_2_ancestral" => new Dpm2Sampler(sigmas, seed, eta: 1.0f),
            "lms" => new LmsSampler(sigmas),
            "dpmpp_2s_ancestral" => new DpmPlusPlus2SAncestralSampler(sigmas, seed),
            "dpmpp_2m" => new DpmPlusPlus2MSampler(sigmas),
            "dpmpp_2m_sde" => new DpmPlusPlus2MSdeSampler(sigmas, seed),
            _ => throw new NotSupportedException(Refusal(name, key)),
        };
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
