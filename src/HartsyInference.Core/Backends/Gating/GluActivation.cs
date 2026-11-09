namespace HartsyInference.Core.Backends.Gating;

/// <summary>The gate function of a <see cref="GluSpec"/>: what the gate half passes through before it multiplies the value half.</summary>
public enum GluActivation
{
    /// <summary><c>g · sigmoid(g)</c> (SwiGLU).</summary>
    Silu,

    /// <summary><c>0.5 · g · (1 + tanh(√(2/π) · (g + 0.044715 · g³)))</c> (GeGLU with the tanh GELU).</summary>
    GeluTanh,

    /// <summary><c>0.5 · g · (1 + erf(g / √2))</c> (GeGLU with the exact GELU).</summary>
    GeluErf,

    /// <summary>SiTU: the gate becomes <c>β · tanh(g / β) · sigmoid(g)</c> and the value is soft-capped to <c>β<sub>lin</sub> · tanh(u / β<sub>lin</sub>)</c>, with β and β<sub>lin</sub> from the <see cref="GluSpec"/>.</summary>
    Situ,
}
