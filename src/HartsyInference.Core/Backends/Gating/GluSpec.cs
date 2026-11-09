namespace HartsyInference.Core.Backends.Gating;

/// <summary>One gated linear unit: <c>out[r, i] = act(gate[r, i]) · value[r, i]</c> over an input whose last axis packs the
/// two halves in the order <see cref="Gate"/> says.</summary>
/// <param name="Activation">The gate function.</param>
/// <param name="Gate">Which half is the gate.</param>
/// <param name="SituBeta">SiTU gate soft-cap β; read only for <see cref="GluActivation.Situ"/>.</param>
/// <param name="SituLinearBeta">SiTU value soft-cap β<sub>lin</sub>; read only for <see cref="GluActivation.Situ"/>.</param>
public readonly record struct GluSpec(GluActivation Activation, GluGateHalf Gate, float SituBeta, float SituLinearBeta)
{
    /// <summary>A spec with the default SiTU betas. Kept as its own overload so assemblies built against the two-field spec still bind.</summary>
    public GluSpec(GluActivation activation, GluGateHalf gate) : this(activation, gate, SituDefaults.Beta, SituDefaults.LinearBeta) { }

    /// <summary>The two-field deconstruction that assemblies built against the original spec bind to.</summary>
    public void Deconstruct(out GluActivation activation, out GluGateHalf gate)
    {
        activation = Activation;
        gate = Gate;
    }

    /// <summary>SwiGLU over <c>[gate | up]</c>.</summary>
    public static GluSpec SwiGlu => new(GluActivation.Silu, GluGateHalf.First);

    /// <summary>GeGLU over <c>[gate | up]</c> with the tanh GELU.</summary>
    public static GluSpec GeGluFirst => new(GluActivation.GeluTanh, GluGateHalf.First);

    /// <summary>GEGLU over <c>[value | gate]</c> with the tanh GELU.</summary>
    public static GluSpec GeGlu => new(GluActivation.GeluTanh, GluGateHalf.Second);

    /// <summary>GEGLU over <c>[value | gate]</c> with the exact GELU.</summary>
    public static GluSpec GeGluErf => new(GluActivation.GeluErf, GluGateHalf.Second);

    /// <summary>SiTU-GLU over <c>[value | gate]</c>.</summary>
    public static GluSpec Situ => new(GluActivation.Situ, GluGateHalf.Second);
}
