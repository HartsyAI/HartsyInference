using HartsyInference.Engine.Audio;

namespace HartsyInference.Engine.Recipes.Music;

/// <summary>ACE-Step 1.5 (and the 1.5 XL line). The per-variant steps, shift and guidance follow upstream's
/// <c>get_ui_control_config</c>: turbo 8 steps, shift 3 (shift 1 on the shift1 checkpoint) at CFG 1; sft 50 and base 32 steps
/// at shift 1. The 120 s duration and turbo's CFG 1 are SwarmUI's (<c>Audio Model Support.md</c>, ACE-Step;
/// <c>WorkflowGenerator.cs</c> defaults the ACE-Step duration to 120).
/// A placed checkpoint is only known to be 1.5 after its header is read, so the loader applies the variant defaults itself.</summary>
public sealed class AceStepMusicRecipe : IMusicRecipe
{
    /// <summary>The 1.5 pipeline's guidance scale for the undistilled sft and base checkpoints.</summary>
    private const double GuidanceScale = 7;

    /// <summary>Turbo's guidance scale: distilled, so unguided; the pipeline also clamps turbo to 1.0.</summary>
    private const double TurboGuidanceScale = 1;

    /// <inheritdoc/>
    public string Name => AudioWeightsCatalog.AceStepId;

    /// <inheritdoc/>
    public bool Matches(string familyId) => string.Equals(familyId, Name, StringComparison.OrdinalIgnoreCase);

    /// <inheritdoc/>
    public MusicDefaults Defaults { get; } = new MusicDefaults
    {
        Revision = 2,
        Duration = 120,
        InferMethod = "ode",
        GuidanceType = "apg",
        CfgIntervalStart = 0,
        CfgIntervalEnd = 1,
        CoverStrength = 0.5,
        Thinking = true,
        LmTemperature = 0.85,
        LmCfgScale = 2,
        LmTopK = 0,
        LmTopP = 0.9,
    };

    /// <inheritdoc/>
    public MusicConstraints Constraints { get; } = new MusicConstraints { MaxDurationSeconds = 600 };

    /// <inheritdoc/>
    public IReadOnlyList<string> Variants { get; } =
    [
        "turbo", "turbo-shift1", "turbo-shift3", "turbo-continuous", "sft", "base", "xl-turbo", "xl-sft", "xl-base",
    ];

    /// <inheritdoc/>
    public MusicDefaults DefaultsFor(string variant)
    {
        (int steps, double shift, double cfgScale) = VariantSampling(variant);
        return Defaults with { Steps = steps, Shift = shift, CfgScale = cfgScale };
    }

    /// <summary>Matches the variant by name, so an id this list does not name still gets its family's turbo values.</summary>
    private static (int Steps, double Shift, double CfgScale) VariantSampling(string variant)
    {
        string value = (variant ?? string.Empty).Trim().ToLowerInvariant();
        if (value.Contains("shift1", StringComparison.Ordinal))
        {
            return (8, 1, TurboGuidanceScale);
        }
        if (value == "sft" || value.EndsWith("-sft", StringComparison.Ordinal))
        {
            return (50, 1, GuidanceScale);
        }
        if (value == "base" || value.EndsWith("-base", StringComparison.Ordinal))
        {
            return (32, 1, GuidanceScale);
        }
        return (8, 3, TurboGuidanceScale);
    }
}
