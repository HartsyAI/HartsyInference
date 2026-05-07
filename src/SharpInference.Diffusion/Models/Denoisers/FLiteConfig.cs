namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for F-Lite (Freepik / Fal.ai) — single-stream cross-attention DiT with T5-XXL conditioning, 16-channel Flux VAE, and 16 register tokens. See [`F_LITE_ARCHITECTURE.md`](../../../../docs/Research/F_LITE_ARCHITECTURE.md) for the architectural deep-dive.</summary>
public record FLiteConfig
{
    /// <summary>Hidden dimension. F-Lite 10B = 3072.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of self-attention / cross-attention heads. F-Lite 10B = 12.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension; computed as <c>HiddenSize / NumHeads</c>. F-Lite 10B = 256.</summary>
    public int HeadDim => HiddenSize / NumHeads;

    /// <summary>Number of transformer blocks. F-Lite 10B = 40.</summary>
    public required int Depth { get; init; }

    /// <summary>Latent channel count (matches Flux VAE).</summary>
    public int InChannels { get; init; } = 16;

    /// <summary>Patch size for latent → token embedding.</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>MLP expansion ratio.</summary>
    public float MlpRatio { get; init; } = 4.0f;

    /// <summary>Inner MLP dimension; computed as <c>HiddenSize * MlpRatio</c>.</summary>
    public int MlpDim => (int)(HiddenSize * MlpRatio);

    /// <summary>Cross-attention input dim (T5-XXL hidden = 4096).</summary>
    public int CrossAttnInputSize { get; init; } = 4096;

    /// <summary>RoPE base frequency.</summary>
    public int RopeBase { get; init; } = 10000;

    /// <summary>Whether the model has trainable bias on linears and a learned RMS scale. F-Lite 10B = false (no biases on QKV/MLP/proj, no learned RMS scale). When true, every Linear has a bias and every RMSNorm has a learned scale weight.</summary>
    public bool TrainBiasAndRms { get; init; } = false;

    /// <summary>Whether self-attention V is mixed across blocks via a learnable lambda residual (block 0's V is fed forward into blocks 1..N as <c>v = lambda * v_current + (1 - lambda) * v_0</c>). F-Lite 10B = true.</summary>
    public bool ResidualV { get; init; } = true;

    /// <summary>Whether to apply 2D RoPE to self-attention Q/K. F-Lite 10B = true.</summary>
    public bool UseRope { get; init; } = true;

    /// <summary>Number of learnable register tokens prepended to image tokens before the first block. F-Lite 10B = 16.</summary>
    public int NumRegisterTokens { get; init; } = 16;

    /// <summary>Maximum precomputed RoPE grid dimension. F-Lite ships with a 512×512 grid (covering up to 1024×1024 image at patch_size=2). Increase if you need higher resolution at runtime.</summary>
    public int RopeMaxGrid { get; init; } = 512;

    /// <summary>T5 hidden state layer index to use for cross-attention conditioning. F-Lite uses index 17 of 24 (Python <c>hidden_states[-8]</c> with <c>final_layer_norm</c> + <c>dropout</c> reapplied).</summary>
    public int T5LayerIndex { get; init; } = 17;

    /// <summary>Default CFG scale.</summary>
    public float DefaultCfgScale { get; init; } = 6.0f;

    /// <summary>Default denoise step count.</summary>
    public int DefaultSteps { get; init; } = 30;

    /// <summary>F-Lite (10B) full-precision preset. Verified against `Freepik/F-Lite/dit_model/config.json` (v0.32.2). 40 blocks × 3072 hidden × 12 heads. Pairs with T5-XXL + Flux Schnell VAE.</summary>
    public static FLiteConfig V1 => new()
    {
        HiddenSize = 3072,
        NumHeads = 12,
        Depth = 40,
        InChannels = 16,
        PatchSize = 2,
        MlpRatio = 4.0f,
        CrossAttnInputSize = 4096,
        RopeBase = 10000,
        TrainBiasAndRms = false,
        ResidualV = true,
        UseRope = true,
        NumRegisterTokens = 16,
        RopeMaxGrid = 512,
    };

    /// <summary>F-Lite-7B distilled preset. Same architectural shape as <see cref="V1"/> per the technical report; the 7B was produced by knowledge distillation from the 10B with reduced depth + width. **Exact numbers held until config.json from the 7B repo is observed**; placeholder values are scaled-down guesses pending verification.</summary>
    public static FLiteConfig V1_7B => new()
    {
        HiddenSize = 2560,
        NumHeads = 10,
        Depth = 32,
        InChannels = 16,
        PatchSize = 2,
        MlpRatio = 4.0f,
        CrossAttnInputSize = 4096,
        RopeBase = 10000,
        TrainBiasAndRms = false,
        ResidualV = true,
        UseRope = true,
        NumRegisterTokens = 16,
        RopeMaxGrid = 512,
    };

    /// <summary>F-Lite-Texture preset (texture-specialized variant). Same backbone as <see cref="V1"/>; the difference is in fine-tune weights, not architecture.</summary>
    public static FLiteConfig Texture => V1;
}
