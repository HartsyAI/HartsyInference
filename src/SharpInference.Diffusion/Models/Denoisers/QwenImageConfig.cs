namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Qwen-Image / Qwen-Image 2.0 MMDiT transformers (7B-20B by Alibaba). Features unified generation and editing (inpaint, outpaint, relighting, style transfer) through a single model. Uses Qwen VL text encoder and MMDiT architecture with AdaLN-Zero modulation.</summary>
public record QwenImageConfig
{
    /// <summary>Hidden dimension of the transformer.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension.</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Number of joint transformer blocks.</summary>
    public required int Depth { get; init; }

    /// <summary>Patch size for latent→token embedding.</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Number of latent channels.</summary>
    public int InChannels { get; init; } = 16;

    /// <summary>Qwen VL text encoder context dimension.</summary>
    public int ContextDim { get; init; } = 4096;

    /// <summary>Pooled projection dimension.</summary>
    public int PooledProjectionDim { get; init; } = 2048;

    /// <summary>MLP ratio for feed-forward blocks.</summary>
    public float MlpRatio { get; init; } = 4.0f;

    /// <summary>RoPE base frequency.</summary>
    public int RopeTheta { get; init; } = 10000;

    /// <summary>QK-norm epsilon.</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    /// <summary>Whether to use QK-norm.</summary>
    public bool UseQkNorm { get; init; } = true;

    /// <summary>Whether this model supports unified editing tasks (inpaint, outpaint, etc.).</summary>
    public bool SupportsEditing { get; init; }

    /// <summary>`Qwen/Qwen-Image` (20B): 60 transformer blocks, hidden=3072, 24 heads, head_dim=128. Matches diffusers `transformer_qwenimage.py` config. Pairs with Qwen2.5-VL-7B text encoder (~15 GB at BF16). Requires ~30 GB total VRAM at FP8 — fits 40+ GB cards (A100 / L40S / 4090 with offload) or needs Q4_K backbone quant for 12 GB consumer cards.</summary>
    public static QwenImageConfig V1 => new()
    {
        HiddenSize = 3072,
        NumHeads = 24,
        Depth = 60,
        SupportsEditing = false,
    };

    /// <summary>Alias kept for back-compat. Identical to <see cref="V1"/>; the original "_7B" suffix was a misnomer — the public Qwen-Image weights are 20B (the "7B" pairs with Qwen2.5-VL-7B as the text encoder, not the diffusion backbone).</summary>
    public static QwenImageConfig V1_7B => V1;

    /// <summary>Speculative Qwen-Image 2.0 14B preset. <b>Not yet released by Alibaba</b> — placeholder values held until an actual checkpoint config is published. Do not rely on these dimensions.</summary>
    public static QwenImageConfig V2_14B => new()
    {
        HiddenSize = 4096,
        NumHeads = 32,
        Depth = 32,
        SupportsEditing = true,
    };

    /// <summary>Speculative Qwen-Image 2.0 20B preset. Same caveat as <see cref="V2_14B"/>.</summary>
    public static QwenImageConfig V2_20B => new()
    {
        HiddenSize = 5120,
        NumHeads = 40,
        Depth = 40,
        SupportsEditing = true,
    };
}
