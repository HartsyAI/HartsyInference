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

    /// <summary>MOSS-Image's SiTU: the gate becomes <c>4 · tanh(g / 4) · sigmoid(g)</c> and the value is soft-capped to <c>25 · tanh(u / 25)</c>.</summary>
    Situ,
}
