namespace HartsyInference.Diffusion.Schedulers;

/// <summary>A named Ideogram 4 sampler preset (upstream <c>sampler_configs.py</c>). Bundles step count, the per-step guidance schedule, and the logit-normal mean/std.
///
/// <see cref="GuidanceSchedule"/> is in **loop-index order**: index 0 is the LAST (polish) step, index <c>NumSteps − 1</c> is the FIRST sampling step. Each preset runs the bulk of steps at gw=7, then a few "polish" steps at gw=3 near <c>t→0</c>. The denoise loop iterates <c>i = NumSteps−1 … 0</c> and reads <c>GuidanceSchedule[i]</c>.
///
/// <see cref="Mu"/> is the base mean passed to <see cref="LogitNormalSchedule.ForResolution"/> as <c>knownMean</c> (before the resolution adjust); <see cref="Std"/> is the logit-normal spread.</summary>
public sealed record Ideogram4SamplerPreset
{
    /// <summary>Preset name (e.g. <c>"V4_QUALITY_48"</c>).</summary>
    public required string Name { get; init; }

    /// <summary>Number of Euler steps.</summary>
    public required int NumSteps { get; init; }

    /// <summary>Per-step guidance weights in loop-index order (index 0 = last/polish step). Length equals <see cref="NumSteps"/>.</summary>
    public required float[] GuidanceSchedule { get; init; }

    /// <summary>Base mean of the logit-normal schedule (before resolution adjust).</summary>
    public required float Mu { get; init; }

    /// <summary>Standard deviation of the logit-normal schedule.</summary>
    public required float Std { get; init; }

    private static float[] Polish(int polishSteps, int mainSteps, float polishGw = 3.0f, float mainGw = 7.0f)
    {
        float[] s = new float[polishSteps + mainSteps];
        for (int i = 0; i < polishSteps; i++) s[i] = polishGw;
        for (int i = polishSteps; i < s.Length; i++) s[i] = mainGw;
        return s;
    }

    /// <summary>48 steps, 3 polish — the default quality preset.</summary>
    public static Ideogram4SamplerPreset Quality48 => new()
    {
        Name = "V4_QUALITY_48",
        NumSteps = 48,
        GuidanceSchedule = Polish(3, 45),
        Mu = 0.0f,
        Std = 1.5f,
    };

    /// <summary>20 steps, 2 polish — balanced default.</summary>
    public static Ideogram4SamplerPreset Default20 => new()
    {
        Name = "V4_DEFAULT_20",
        NumSteps = 20,
        GuidanceSchedule = Polish(2, 18),
        Mu = 0.0f,
        Std = 1.75f,
    };

    /// <summary>12 steps, 1 polish — fast preset.</summary>
    public static Ideogram4SamplerPreset Turbo12 => new()
    {
        Name = "V4_TURBO_12",
        NumSteps = 12,
        GuidanceSchedule = Polish(1, 11),
        Mu = 0.5f,
        Std = 1.75f,
    };

    /// <summary>Looks up a preset by name (case-insensitive). Throws if unknown.</summary>
    public static Ideogram4SamplerPreset ByName(string name) => name.ToUpperInvariant() switch
    {
        "V4_QUALITY_48" => Quality48,
        "V4_DEFAULT_20" => Default20,
        "V4_TURBO_12" => Turbo12,
        _ => throw new ArgumentException($"Unknown Ideogram 4 sampler preset '{name}'. Valid: V4_QUALITY_48, V4_DEFAULT_20, V4_TURBO_12.", nameof(name)),
    };
}
