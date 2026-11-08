namespace HartsyInference.Core.Backends.Gating;

/// <summary>One gated linear unit: <c>out[r, i] = act(gate[r, i]) · value[r, i]</c> over an input whose last axis packs the
/// two halves in the order <see cref="Gate"/> says.</summary>
/// <param name="Activation">The gate function.</param>
/// <param name="Gate">Which half is the gate.</param>
public readonly record struct GluSpec(GluActivation Activation, GluGateHalf Gate)
{
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
