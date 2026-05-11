namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Anima (CircleStone Labs / Comfy Org, 2B params, derivative of NVIDIA's Cosmos-Predict2-2B-Text2Image).
/// Verified against <c>diffusers/models/transformers/transformer_cosmos.py</c> (<c>CosmosTransformer3DModel</c>) and the published
/// <c>nvidia/Cosmos-Predict2-2B-Text2Image</c> + <c>circlestone-labs/Anima</c> distributions.
///
/// **Architecture:** <see cref="AnimaTransformer"/> is a 3D DiT designed for video, but for the t2i variant the temporal axis
/// collapses to a single frame (<c>T = 1</c>, <c>p_t = 1</c>). All video-specific code paths (FPS-aware RoPE scaling, multi-frame
/// patchification, expanding per-frame timesteps over spatial dims) are dropped — this implementation is image-only and
/// will throw if the latent has more than one temporal slice. Notable architectural details:
/// <list type="bullet">
///   <item>Patch embedding is a single <c>Linear</c> over the flattened patch — <c>(in_channels * p_t * p_h * p_w) → hidden</c>.
///        For Cosmos2 t2i: <c>16 × 1 × 2 × 2 = 64 → 2048</c>. There is also an optional concat-padding-mask channel that
///        bumps <c>in_channels</c> to <c>17</c> during the linear projection (<c>concat_padding_mask=True</c>).</item>
///   <item>3-axis RoPE on <c>(t, h, w)</c> with NTK frequency scaling per axis (<c>rope_scale=(2.0, 1.0, 1.0)</c>). For t2i
///        with <c>T = 1</c> the temporal sub-RoPE collapses to angle 0 (cos=1, sin=0) and is a no-op.</item>
///   <item>Optional **learnable** positional embedding (3 separate parameters for t/h/w that are summed and L2-normalized).
///        Anima ships this (<c>extra_pos_embed_type="learnable"</c>).</item>
///   <item>Time embedding: <c>Timesteps(sin/cos, flip_sin_to_cos=True) → CosmosTimestepEmbedding (Linear→SiLU→Linear)</c>
///        produces <c>temb [B, condition_dim]</c>. The pre-MLP sinusoidal vector is also fed through an <c>RMSNorm</c> to
///        produce <c>embedded_timestep [B, hidden]</c>. Both vectors are passed into every block's AdaLN.</item>
///   <item>Each block has THREE AdaLN-Zero modulators — one per sub-block (self-attn, cross-attn, FFN). Each <c>AdaLN_Zero</c>
///        emits <c>(shift, scale, gate)</c> via <c>Linear(silu(embedded_timestep)) + Linear → +temb[:3*hidden] → chunk(3, -1)</c>.</item>
///   <item>**Self-attention** with QK-RMSNorm and 3D RoPE on Q+K. Standard SDPA, no joint context.</item>
///   <item>**Cross-attention**: Q from hidden, K/V from text encoder hidden states. Optional <c>crossattn_proj</c> projects
///        text features to hidden (Anima uses <c>use_crossattn_projection=True</c> when text encoder dim != hidden).</item>
///   <item>FFN: standard <c>Linear → GELU → Linear</c>, <c>mlp_ratio=4.0</c>.</item>
///   <item>Final layer: <c>CosmosAdaLayerNorm (Linear → +temb[:2*hidden] → chunk(2, -1) → x*(1+scale)+shift)</c>, then
///        <c>proj_out: Linear(hidden, p_t*p_h*p_w * out_channels)</c>, then unpatchify back to <c>[B, C_out, H, W]</c>.</item>
/// </list>
///
/// **Anima text encoder/VAE (per the model card):** Qwen-3 0.6B base (hidden=1024), Qwen-Image VAE (16-channel latent,
/// 8× spatial downscale). Cosmos-Predict2-2B uses a T5-XXL encoder (hidden=1024 too — same width, so the same DiT can
/// accept either). Anima keeps the Cosmos-Predict2 DiT architecture verbatim and just retrains weights for anime + swaps
/// the text encoder family.
///
/// **What we drop:** all video / world-model paths. <c>num_frames</c> is enforced to 1; <c>fps</c> is unused;
/// <c>condition_mask</c> (Video2World autoregressive conditioning), ControlNet residuals, and <c>image_context</c>
/// (the Predict2.5 hybrid attention path) are not implemented.
/// </summary>
public sealed record AnimaConfig
{
    /// <summary>Hidden dim (= num_attention_heads * head_dim). 2048 for Cosmos-Predict2 2B / Anima.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of attention heads. 16 for Cosmos-Predict2 2B (head_dim 128).</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension. 128 for Cosmos-Predict2 2B (16 × 128 = 2048).</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Number of identical transformer blocks. 28 for Cosmos-Predict2 2B / Anima.</summary>
    public required int NumLayers { get; init; }

    /// <summary>FFN expansion ratio (mlp_ratio). 4.0 in upstream Cosmos. <c>FfnHiddenSize = HiddenSize * MlpRatio</c>.</summary>
    public float MlpRatio { get; init; } = 4.0f;

    /// <summary>Latent input channels (delivered by the Qwen-Image VAE). 16 for Anima.</summary>
    public int InChannels { get; init; } = 16;

    /// <summary>Output channels (back to VAE latent space). 16 for Anima.</summary>
    public int OutChannels { get; init; } = 16;

    /// <summary>3D patch size <c>(p_t, p_h, p_w)</c>. Cosmos default is <c>(1, 2, 2)</c>; for t2i <c>p_t</c> is irrelevant.</summary>
    public (int T, int H, int W) PatchSize { get; init; } = (1, 2, 2);

    /// <summary>Maximum supported latent grid <c>(T, H, W)</c> in tokens (post-patch). Drives the RoPE / learnable-pos-embed
    /// table sizes. Cosmos default <c>(128, 240, 240)</c> with patch <c>(1,2,2)</c> → max <c>(128, 120, 120)</c> tokens.</summary>
    public (int T, int H, int W) MaxSize { get; init; } = (128, 240, 240);

    /// <summary>RoPE NTK frequency scaling per axis <c>(t, h, w)</c>. Cosmos default <c>(2.0, 1.0, 1.0)</c>.</summary>
    public (float T, float H, float W) RopeScale { get; init; } = (2.0f, 1.0f, 1.0f);

    /// <summary>RoPE base period before NTK scaling. 10000 for Cosmos.</summary>
    public float RopeTheta { get; init; } = 10000.0f;

    /// <summary>Reference base FPS for temporal RoPE scaling. Image-only mode ignores this; we keep the field for
    /// future video reuse and to match the upstream config so weights load cleanly.</summary>
    public int BaseFps { get; init; } = 16;

    /// <summary>Whether to concatenate a constant-1 padding-mask channel to the latent before the patch embedding.
    /// True in stock Cosmos2; the patch_embed weight then expects <c>(in_channels + 1) * p_t * p_h * p_w</c> input dim.</summary>
    public bool ConcatPaddingMask { get; init; } = true;

    /// <summary>Whether the model owns a learnable additive positional embedding (Cosmos's
    /// <c>extra_pos_embed_type == "learnable"</c>). True for Cosmos2-T2I / Anima.</summary>
    public bool UseLearnablePosEmbed { get; init; } = true;

    /// <summary>Whether the cross-attention path applies a learned projection to the text encoder hidden states
    /// before they hit K/V. True when text encoder hidden ≠ <see cref="HiddenSize"/>.</summary>
    public bool UseCrossAttnProjection { get; init; } = true;

    /// <summary>Text encoder output dim (<c>encoder_hidden_states_channels</c> in Cosmos). Qwen-3 0.6B base = 1024;
    /// Cosmos2 stock T5-XXL also = 1024 — both work.</summary>
    public int TextEmbedDim { get; init; } = 1024;

    /// <summary>AdaLN-LoRA rank (<c>adaln_lora_dim</c>). Cosmos uses 256 — but our t2i implementation uses the standard
    /// non-LoRA AdaLN path (full <c>Linear(condition_dim, 3*hidden)</c>), so this is informational. The checkpoint may
    /// or may not contain LoRA-style decomposed weights; the converter handles both.</summary>
    public int AdaLnLoraDim { get; init; } = 256;

    /// <summary>Conditioning dim for time embedding (<c>condition_dim</c> in <c>CosmosEmbedding</c>). Equals
    /// <see cref="HiddenSize"/> in stock Cosmos2.</summary>
    public int ConditionDim { get; init; } = 2048;

    /// <summary>RMSNorm epsilon (used for the timestep RMSNorm and any model-internal RMSNorms).</summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>QK-norm epsilon.</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    /// <summary>Cosmos-Predict2-2B-Text2Image preset (matches the upstream <c>_PREDICT2_TEXT2IMAGE_PIPELINE_2B</c> defaults
    /// and the diffusers' <c>CosmosTransformer3DModel</c> 2B config). 28 layers × 2048 hidden, 16 heads of 128 each.
    /// 16-channel latent input, 4×4 effective patch (8× VAE × 2×2 patch = 16× total downscale).</summary>
    public static AnimaConfig Cosmos2_2B => new()
    {
        HiddenSize = 2048,
        NumHeads = 16,
        HeadDim = 128,
        NumLayers = 28,
        MlpRatio = 4.0f,
        InChannels = 16,
        OutChannels = 16,
        PatchSize = (1, 2, 2),
        MaxSize = (128, 240, 240),
        RopeScale = (2.0f, 1.0f, 1.0f),
        RopeTheta = 10000.0f,
        BaseFps = 16,
        ConcatPaddingMask = true,
        UseLearnablePosEmbed = true,
        UseCrossAttnProjection = true,
        TextEmbedDim = 1024,
        AdaLnLoraDim = 256,
        ConditionDim = 2048,
        RmsNormEps = 1e-6f,
        QkNormEps = 1e-6f,
    };

    /// <summary>Anima Preview-3 base preset. Same DiT geometry as <see cref="Cosmos2_2B"/>; only the trained weights and
    /// the recommended text encoder differ (Anima uses Qwen-3 0.6B base instead of T5-XXL). Both encoders happen to
    /// produce <c>1024</c>-dim hiddens so <see cref="TextEmbedDim"/> is unchanged.</summary>
    public static AnimaConfig AnimaPreview3 => Cosmos2_2B;

    /// <summary>Convenience: the FFN inner dimension after applying <see cref="MlpRatio"/>.</summary>
    public int FfnHiddenSize => (int)(HiddenSize * MlpRatio);

    /// <summary>Convenience: total flat patch dimension <c>p_t * p_h * p_w</c>.</summary>
    public int PatchVolume => PatchSize.T * PatchSize.H * PatchSize.W;
}
