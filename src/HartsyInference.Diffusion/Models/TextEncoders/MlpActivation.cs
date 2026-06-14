namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>MLP activation function variant. <see cref="Silu"/> matches Llama / Qwen / Mistral SwiGLU; <see cref="GeluTanh"/> matches Gemma 2's GeGLU (<c>gelu(x, approximate="tanh")</c> in PyTorch).</summary>
public enum MlpActivation
{
    /// <summary>SiLU/Swish: <c>x * sigmoid(x)</c>. Default for Llama family.</summary>
    Silu,

    /// <summary>Tanh-approximation GELU (Gemma 2 family).</summary>
    GeluTanh,
}
