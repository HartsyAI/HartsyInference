namespace HartsyInference.Core.Moe;

/// <summary>
/// The per-expert computation as data: <c>down( act(clampGate(gate(x))) * clampUp(up(x)) )</c>. The runtime executes
/// programs; it never hard-codes <c>down(silu(gate(x)) * up(x))</c>. Clamp bounds are independent because some models
/// clamp the two branches asymmetrically (DeepSeek-V4.1 caps the gate from above only and clamps up to ±limit).
/// </summary>
/// <param name="Activation">Gate activation.</param>
/// <param name="GateMax">Upper bound on the gate pre-activation; <see cref="float.PositiveInfinity"/> disables it.</param>
/// <param name="UpMin">Lower bound on the up branch; <see cref="float.NegativeInfinity"/> disables it.</param>
/// <param name="UpMax">Upper bound on the up branch; <see cref="float.PositiveInfinity"/> disables it.</param>
public sealed record ExpertProgram(ExpertActivation Activation, float GateMax, float UpMin, float UpMax)
{
    /// <summary>Rejects NaN bounds and an inverted up range, which would make <see cref="Clamp"/> throw.</summary>
    /// <exception cref="ArgumentException">A bound is NaN or <see cref="UpMin"/> exceeds <see cref="UpMax"/>.</exception>
    public ExpertProgram Validated()
    {
        if (float.IsNaN(GateMax) || float.IsNaN(UpMin) || float.IsNaN(UpMax)) throw new ArgumentException("Clamp bounds must not be NaN.");
        if (UpMin > UpMax) throw new ArgumentException($"Up clamp range is inverted: min {UpMin} exceeds max {UpMax}.");
        return this;
    }

    /// <summary>Plain SwiGLU with no clamp.</summary>
    public static ExpertProgram Swiglu { get; } = new(ExpertActivation.Silu, float.PositiveInfinity, float.NegativeInfinity, float.PositiveInfinity);

    /// <summary>GeGLU with no clamp.</summary>
    public static ExpertProgram GeGlu { get; } = new(ExpertActivation.GeluTanh, float.PositiveInfinity, float.NegativeInfinity, float.PositiveInfinity);

    /// <summary>SwiGLU with the DeepSeek-V4.1 clamp: gate capped above at <paramref name="limit"/>, up clamped to ±<paramref name="limit"/>.</summary>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="limit"/> is not positive and finite.</exception>
    public static ExpertProgram SwigluClamped(float limit)
    {
        if (!(limit > 0f) || float.IsInfinity(limit)) throw new ArgumentOutOfRangeException(nameof(limit), limit, "The SwiGLU clamp limit must be positive and finite.");
        return new ExpertProgram(ExpertActivation.Silu, limit, -limit, limit);
    }

    /// <summary>True when any bound is finite.</summary>
    public bool IsClamped => !float.IsPositiveInfinity(GateMax) || !float.IsNegativeInfinity(UpMin) || !float.IsPositiveInfinity(UpMax);

    /// <summary>Applies the clamps to one token's gate and up pre-activations, returning the clamped pair.</summary>
    public (float Gate, float Up) Clamp(float gate, float up)
    {
        float g = MathF.Min(gate, GateMax);
        float u = Math.Clamp(up, UpMin, UpMax);
        return (g, u);
    }
}
