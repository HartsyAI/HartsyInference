namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Qwen-Image / Qwen-Image 2.0 MMDiT transformers (7B-20B by Alibaba). Features unified generation and editing (inpaint, outpaint, relighting, style transfer) through a single model. Uses Qwen VL text encoder and MMDiT architecture with AdaLN-Zero modulation.</summary>
public sealed record QwenImageConfig
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

    /// <summary>Qwen VL text encoder context dimension. Qwen-Image's `joint_attention_dim` is <b>3584</b>
    /// (= Qwen2.5-VL-7B hidden size), NOT 4096 — the text in-projection weight is shaped against this, so it
    /// must match the encoder hidden dim or the projection will mis-shape on load.</summary>
    public int ContextDim { get; init; } = 3584;

    /// <summary>Pooled projection dimension (`pooled_projection_dim` = 768 in the diffusers config). Qwen-Image's
    /// MMDiT does not use a pooled-CLIP conditioning path the way SD3/Flux do, so this is largely vestigial, but
    /// the value mirrors the reference config.</summary>
    public int PooledProjectionDim { get; init; } = 768;

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

    /// <summary>Whether text tokens receive RoPE. Qwen-Image ropes both streams (true); Mage-Flow sets
    /// <c>apply_text_rotary_emb=false</c> — text tokens get identity rotation, only image tokens are rotated.</summary>
    public bool RopeText { get; init; } = true;

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

    /// <summary>Microsoft <b>Mage-Flow</b> (4B NR-MMDiT, arXiv 2607.19064). Same MMDiT block/attention convention as
    /// Qwen-Image (diffusers <c>to_q</c>/<c>add_q_proj</c>/<c>norm_added_q</c>/<c>to_add_out</c>, timestep-only AdaLN,
    /// gelu MLP, 3-axis RoPE <c>[16,56,56]</c> θ=10000) — so it reuses <see cref="QwenImageTransformer"/>/
    /// <c>QwenImageBlock</c> verbatim. Differences captured here: <b>12</b> dual-stream blocks (no single-stream
    /// stage), <b>patch_size 1</b> (all 16× reduction is in MageVAE, so no patchify), <b>128</b> latent channels,
    /// and the text encoder is Qwen3-VL-4B (<c>context_in_dim 2560</c>; the <c>txt_in</c> Linear 2560→3072 loads from
    /// the checkpoint). RoPE is applied to IMAGE tokens only (<c>apply_text_rotary_emb=false</c>) — see the
    /// image-only path in the Mage-Flow pipeline. Scheduler: FlowMatchEuler static shift 6.0 (Z-Image schedule).</summary>
    public static QwenImageConfig MageFlow => new()
    {
        HiddenSize = 3072,
        NumHeads = 24,
        Depth = 12,
        PatchSize = 1,
        InChannels = 128,
        ContextDim = 2560,   // Qwen3-VL-4B hidden size
        SupportsEditing = true,   // Mage-Flow-Edit-Turbo reuses this DiT with reference-latent tokens
        RopeText = false,         // apply_text_rotary_emb=false — RoPE on image tokens only
    };

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
