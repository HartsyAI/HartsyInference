namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Flux.2 transformers (Flux.2 Dev 32B, Flux.2 Klein 4B/9B). Architecturally distinct from Flux.1: LayerNorm (not RMSNorm) for stream norms, top-level shared modulation projections (one set per stream type, not per-block), 4-axis RoPE (T,H,W,L) with theta=2000, parallel single-stream block (fused QKV+MLP linear1, fused output linear2), SwiGLU MLP, 32-ch VAE latent + 2×2 patchify giving 128 in-channels per token. Reference: huggingface/diffusers transformer_flux2.py.</summary>
public sealed record Flux2Config
{
    /// <summary>Hidden dimension (= NumHeads × HeadDim). 3072 for Klein 4B, 4096 for Klein 9B, 6144 for Dev.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension. Always 128 for Flux.2 variants. Equals sum(AxesDim).</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Number of double-stream (img+txt joint attention) blocks.</summary>
    public required int Depth { get; init; }

    /// <summary>Number of single-stream (parallel attn+MLP) blocks.</summary>
    public required int DepthSingleBlocks { get; init; }

    /// <summary>Input channels per packed patch token. 32-ch VAE latent × 2×2 patchify = 128.</summary>
    public int InChannels { get; init; } = 128;

    /// <summary>Output channels per packed patch token (matches InChannels).</summary>
    public int OutChannels { get; init; } = 128;

    /// <summary>Text encoder hidden dimension into the transformer's context_embedder. Klein concatenates 3 Qwen3-4B layers → 3 × 2560 = 7680. Klein 9B uses larger Qwen variant. Dev uses Mistral-Small-3.</summary>
    public int ContextInDim { get; init; } = 7680;

    /// <summary>MLP ratio for SwiGLU FFN (mlp_inner = HiddenSize × MlpRatio).</summary>
    public float MlpRatio { get; init; } = 3.0f;

    /// <summary>RoPE per-axis dimensions (T, H, W, L). Sums to HeadDim=128.</summary>
    public int[] AxesDim { get; init; } = [32, 32, 32, 32];

    /// <summary>RoPE base frequency. 2000 for Flux.2 (vs 10000 for Flux.1).</summary>
    public int Theta { get; init; } = 2000;

    /// <summary>Whether QKV projections have bias (false for all Flux.2 variants).</summary>
    public bool QkvBias { get; init; } = false;

    /// <summary>Whether to embed guidance scale via MLP. True for Dev (CFG-distilled), false for Klein (no CFG).</summary>
    public bool GuidanceEmbed { get; init; } = false;

    /// <summary>QK-norm epsilon (RMSNorm on per-head Q/K, pre-RoPE).</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    /// <summary>LayerNorm epsilon for stream norms (norm1, norm2 in double; norm in single).</summary>
    public float LayerNormEps { get; init; } = 1e-6f;

    /// <summary>VAE downsample factor (image_size / latent_size). Flux.2 VAE is 8×, then 2×2 patchify on top → 16× effective for transformer tokens.</summary>
    public int VaeDownscaleFactor { get; init; } = 16;

    /// <summary>VAE latent channel count (pre-patchify).</summary>
    public int VaeLatentChannels { get; init; } = 32;

    /// <summary>Patch size used to repack latent into transformer tokens.</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Sinusoidal timestep embedding input channel count (256 for all Flux.2 variants).</summary>
    public int TimestepChannels { get; init; } = 256;

    /// <summary>Text encoder type used by this variant.</summary>
    public required Flux2TextEncoderType TextEncoderType { get; init; }

    /// <summary>Flux.2 Klein 4B preset. 5 double + 20 single, hidden=3072, 24 heads. Verified against Comfy-Org/flux2-klein/flux-2-klein-4b.safetensors.</summary>
    public static Flux2Config Klein4B => new()
    {
        HiddenSize = 3072,
        NumHeads = 24,
        HeadDim = 128,
        Depth = 5,
        DepthSingleBlocks = 20,
        InChannels = 128,
        OutChannels = 128,
        ContextInDim = 7680,
        MlpRatio = 3.0f,
        AxesDim = [32, 32, 32, 32],
        Theta = 2000,
        QkvBias = false,
        GuidanceEmbed = false,
        TextEncoderType = Flux2TextEncoderType.Qwen,
    };

    /// <summary>Flux.2 Klein 9B preset. Documented architecture: 8 double + 24 single, hidden=4096, 32 heads, joint_attention_dim=12288. Not yet verified against checkpoint.</summary>
    public static Flux2Config Klein9B => new()
    {
        HiddenSize = 4096,
        NumHeads = 32,
        HeadDim = 128,
        Depth = 8,
        DepthSingleBlocks = 24,
        InChannels = 128,
        OutChannels = 128,
        ContextInDim = 12288,
        MlpRatio = 3.0f,
        AxesDim = [32, 32, 32, 32],
        Theta = 2000,
        QkvBias = false,
        GuidanceEmbed = false,
        TextEncoderType = Flux2TextEncoderType.Qwen,
    };

    /// <summary>Flux.2 Dev 32B preset (defaults from diffusers Flux2Transformer2DModel). 8 double + 48 single, hidden=6144, 48 heads, Mistral text encoder, guidance embedding for CFG distillation.</summary>
    public static Flux2Config Dev => new()
    {
        HiddenSize = 6144,
        NumHeads = 48,
        HeadDim = 128,
        Depth = 8,
        DepthSingleBlocks = 48,
        InChannels = 128,
        OutChannels = 128,
        ContextInDim = 15360,
        MlpRatio = 3.0f,
        AxesDim = [32, 32, 32, 32],
        Theta = 2000,
        QkvBias = false,
        GuidanceEmbed = true,
        TextEncoderType = Flux2TextEncoderType.Mistral,
    };
}

/// <summary>Text encoder type used by Flux.2 variants.</summary>
public enum Flux2TextEncoderType
{
    /// <summary>Mistral-Small-3 used by Flux.2 Dev 32B.</summary>
    Mistral,

    /// <summary>Qwen3-4B used by Flux.2 Klein 4B/9B (multi-layer hidden state concat).</summary>
    Qwen,
}
