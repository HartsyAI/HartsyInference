namespace HartsyInference.Core.Moe;

/// <summary>Gate activation of an expert FFN. Up is always a linear branch (gated FFN); a non-gated expert is not modeled yet.</summary>
public enum ExpertActivation
{
    /// <summary>SiLU gate (SwiGLU): Qwen-MoE, Mixtral, DeepSeek.</summary>
    Silu,

    /// <summary>Tanh-approximated GELU gate (GeGLU): Gemma-style MoE layers.</summary>
    GeluTanh,

    /// <summary>ReLU gate.</summary>
    Relu,

    /// <summary>Squared ReLU gate.</summary>
    ReluSquared,
}
