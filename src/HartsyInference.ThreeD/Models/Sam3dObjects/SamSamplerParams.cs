namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>Parameters of one SAM 3D flow-matching run: Euler steps over <c>t = 0 → 1</c> with a time warp and a guidance interval.</summary>
public sealed record SamSamplerParams
{
    /// <summary>Euler steps (<c>inference_steps</c>).</summary>
    public required int Steps { get; init; }

    /// <summary>Time-warp factor of the schedule (<c>rescale_t</c>); 1 is the linear grid.</summary>
    public required double RescaleT { get; init; }

    /// <summary>Guidance strength <c>s</c> of <c>(1 + s)·cond − s·uncond</c> (<c>strength</c>); 0 runs the conditional pass alone.</summary>
    public required float Strength { get; init; }

    /// <summary>Guidance applies while <c>low &lt;= 1000·t &lt;= high</c>, in the model's scaled time (<c>interval</c>).</summary>
    public double IntervalLow { get; init; }

    public double IntervalHigh { get; init; } = 500.0;

    /// <summary>Factor between the schedule time and the value the model sees (<c>time_scale</c>).</summary>
    public double TimeScale { get; init; } = 1000.0;

    /// <summary>Shortcut step size fed to the model, already in model units; 0 for plain sampling (<c>no_shortcut</c>).</summary>
    public double ShortcutStep { get; init; }

    /// <summary>The structure stage as <c>pipeline.yaml</c> leaves it: 25 steps, <c>rescale_t</c> 3, strength 7, interval [0, 500].</summary>
    public static SamSamplerParams StructureDefault { get; } = new() { Steps = 25, RescaleT = 3.0, Strength = 7f };

    /// <summary>The SLAT stage as <c>pipeline.yaml</c> sets it: 25 steps, <c>rescale_t</c> 1, strength 1, interval [0, 500].</summary>
    public static SamSamplerParams SlatDefault { get; } = new() { Steps = 25, RescaleT = 1.0, Strength = 1f };

    /// <summary>Throws <see cref="ArgumentException"/> when the parameters cannot drive a sampler.</summary>
    public void Validate()
    {
        if (Steps <= 0)
            throw new ArgumentException($"Sampler steps must be positive; got {Steps}.");
        if (!(RescaleT > 0.0) || !double.IsFinite(RescaleT))
            throw new ArgumentException($"rescale_t must be positive; got {RescaleT}.");
        if (!float.IsFinite(Strength) || Strength < 0f)
            throw new ArgumentException($"Guidance strength must be finite and non-negative; got {Strength}.");
        if (!(TimeScale > 0.0) || !double.IsFinite(TimeScale))
            throw new ArgumentException($"time_scale must be positive; got {TimeScale}.");
        if (!double.IsFinite(IntervalLow) || !double.IsFinite(IntervalHigh) || IntervalLow > IntervalHigh)
            throw new ArgumentException($"Guidance interval [{IntervalLow}, {IntervalHigh}] is empty.");
        if (!double.IsFinite(ShortcutStep))
            throw new ArgumentException($"The shortcut step must be finite; got {ShortcutStep}.");
    }
}
