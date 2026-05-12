namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Anima (CircleStone Labs / Comfy Org, 2B params, derivative of NVIDIA's
/// Cosmos-Predict2-2B-Text2Image). Verified by probing the actual ComfyUI single-file checkpoint
/// <c>anima-preview3-base.safetensors</c> — every tensor name + shape below is measured, not assumed.
///
/// **Architecture (measured):** 28-block DiT operating on a 16-channel 8× VAE latent. All weights live
/// under the <c>net.</c> prefix in the single-file checkpoint.
/// <list type="bullet">
///   <item><c>net.x_embedder.proj.1.weight [2048, 68]</c> — patch embed. <c>68 = (16 in_ch + 1 padding mask) × 2 × 2 patch</c>.
///        Wrapped in <c>nn.Sequential([Identity, Linear])</c>, hence the <c>.1.</c>. No bias.</item>
///   <item><c>net.t_embedder.1.linear_1.weight [2048, 2048]</c> + <c>linear_2.weight [6144, 2048]</c> — time embedder.
///        Sin/cos timesteps → Linear → SiLU → Linear produces <c>temb [B, 6144]</c> (= 3 × hidden). No bias.</item>
///   <item><c>net.t_embedding_norm.weight [2048]</c> — RMSNorm of the pre-MLP sin/cos vector → <c>embedded_timestep [B, 2048]</c>.
///        Both <c>temb [B, 6144]</c> and <c>embedded_timestep [B, 2048]</c> are passed into every block's AdaLN-LoRA.</item>
///   <item>Per block <c>net.blocks.{i}</c> (i ∈ 0..27): three AdaLN-LoRA modulators (one per sub-block: self-attn, cross-attn, MLP),
///        each with <c>adaln_modulation_X.1.weight [256, 2048]</c> + <c>.2.weight [6144, 256]</c>. The full AdaLN flow is
///        <c>silu(embedded_timestep) → Linear → 256 → Linear → 6144</c>, then <c>+ temb[:6144]</c>, then chunk(3) →
///        (shift, scale, gate) each <c>[B, 2048]</c>. No biases anywhere.</item>
///   <item>Self-attention: <c>{q,k,v}_proj.weight [2048, 2048]</c>, <c>output_proj.weight [2048, 2048]</c>, no biases.
///        <c>{q,k}_norm.weight [128]</c> — per-head_dim RMSNorm (16 heads × 128 = 2048).</item>
///   <item>Cross-attention: <c>q_proj.weight [2048, 2048]</c>; <c>{k,v}_proj.weight [2048, 1024]</c> — K/V come from the
///        <c>llm_adapter</c> output (1024-dim). <c>output_proj.weight [2048, 2048]</c>, no biases.</item>
///   <item>MLP: <c>mlp.layer1.weight [8192, 2048]</c>, <c>mlp.layer2.weight [2048, 8192]</c>, no biases. Activation is GELU.</item>
///   <item>Final layer <c>net.final_layer</c>: AdaLN-LoRA <c>.adaln_modulation.1.weight [256, 2048]</c> + <c>.2.weight [4096, 256]</c> →
///        <c>+ temb[:4096]</c> → chunk(2) (shift, scale, no gate). Then <c>final_layer.linear.weight [64, 2048]</c> projects
///        each token to <c>p_h × p_w × out_channels = 4 × 16 = 64</c>. No biases.</item>
/// </list>
///
/// **LLM Adapter (measured, Anima-specific):** A 6-block transformer that sits between the Qwen-3 0.6B text encoder
/// and the DiT cross-attention. Each block has self-attn + cross-attn + MLP, RMSNorm pre-norms.
/// <list type="bullet">
///   <item>Top-level: <c>embed.weight [32128, 1024]</c> (the T5 vocab size — likely an initialization artifact carried over
///        from the original Cosmos-Predict2 training; semantic role to verify by ablation),
///        <c>norm.weight [1024]</c>, <c>out_proj.weight [1024, 1024]</c> + <c>out_proj.bias [1024]</c>.</item>
///   <item>Per block: <c>norm_{self_attn,cross_attn,mlp}.weight [1024]</c>; self+cross-attn each
///        <c>{q,k,v,o}_proj.weight [1024, 1024]</c> (note <c>o_proj</c>, not <c>output_proj</c>) with
///        <c>{q,k}_norm.weight [64]</c> (16 heads × 64 head_dim); MLP <c>mlp.0.weight [4096, 1024]</c> + bias [4096],
///        <c>mlp.2.weight [1024, 4096]</c> + bias [1024]. No RoPE in the adapter.</item>
/// </list>
///
/// **What we drop:** all video / world-model paths (Cosmos's video2world conditioning, ControlNet residuals,
/// hybrid attention). Anima is t2i-only with a single 1024×1024-ish latent grid; <c>num_frames</c> is enforced to 1.
/// </summary>
public sealed record AnimaConfig
{
    /// <summary>Hidden dim of the DiT trunk (= NumHeads × HeadDim). 2048 for Cosmos-Predict2 2B / Anima.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of self-attention heads in the DiT. 16 for Cosmos-Predict2 2B (head_dim 128).</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dim of the DiT. 128 for Cosmos-Predict2 2B (16 × 128 = 2048).</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Number of stacked DiT blocks. 28 for Cosmos-Predict2 2B / Anima.</summary>
    public required int NumLayers { get; init; }

    /// <summary>FFN expansion ratio. 4.0 in upstream Cosmos. <c>FfnHiddenSize = HiddenSize × MlpRatio</c>.</summary>
    public float MlpRatio { get; init; } = 4.0f;

    /// <summary>Latent input channels (delivered by the Qwen-Image VAE). 16 for Anima.</summary>
    public int InChannels { get; init; } = 16;

    /// <summary>Output channels (back to VAE latent space). 16 for Anima.</summary>
    public int OutChannels { get; init; } = 16;

    /// <summary>3D patch size <c>(p_t, p_h, p_w)</c>. Cosmos default <c>(1, 2, 2)</c>; for t2i <c>p_t = 1</c>.</summary>
    public (int T, int H, int W) PatchSize { get; init; } = (1, 2, 2);

    /// <summary>Maximum supported latent grid <c>(T, H, W)</c> in tokens (post-patch). Drives RoPE table sizing.</summary>
    public (int T, int H, int W) MaxSize { get; init; } = (128, 240, 240);

    /// <summary>RoPE NTK frequency scaling per axis. Cosmos default <c>(2.0, 1.0, 1.0)</c>.</summary>
    public (float T, float H, float W) RopeScale { get; init; } = (2.0f, 1.0f, 1.0f);

    /// <summary>RoPE base period before NTK scaling.</summary>
    public float RopeTheta { get; init; } = 10000.0f;

    /// <summary>Whether to concatenate a constant-1 padding-mask channel to the latent before patch embed.
    /// True in Anima (the <c>x_embedder.proj.1.weight</c> expects 17 input channels = 16 latent + 1 mask).</summary>
    public bool ConcatPaddingMask { get; init; } = true;

    /// <summary>Conditioning dim of the time embedder MLP output (<c>t_embedder.1.linear_2.weight [6144, 2048]</c>).
    /// <b>Measured value: 6144 (= 3 × hidden).</b> This is the width of <c>temb</c> consumed by every AdaLN-LoRA
    /// (which adds <c>temb[:6144]</c> for the 3-chunk block AdaLN and <c>temb[:4096]</c> for the 2-chunk final layer).</summary>
    public int ConditionDim { get; init; } = 6144;

    /// <summary>Width of the RMSNorm'd sin/cos vector consumed by the AdaLN-LoRA bottleneck input. <b>Measured: 2048</b>.</summary>
    public int EmbeddedTimestepDim { get; init; } = 2048;

    /// <summary>AdaLN-LoRA bottleneck rank. <b>Measured: 256</b>. The block AdaLN flow is
    /// <c>Linear[256, 2048] → Linear[6144, 256]</c>; the final-layer AdaLN flow is
    /// <c>Linear[256, 2048] → Linear[4096, 256]</c>.</summary>
    public int AdaLnLoraDim { get; init; } = 256;

    /// <summary>RMSNorm epsilon.</summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>QK-norm epsilon.</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    /// <summary>Configuration for the in-checkpoint <c>llm_adapter</c> sub-transformer (Anima-specific).
    /// See <see cref="AnimaConfig"/> remarks for the measured weight layout.</summary>
    public AnimaLlmAdapterConfig LlmAdapter { get; init; } = AnimaLlmAdapterConfig.Default;

    /// <summary>Cosmos-Predict2-2B-Text2Image preset, matching the measured Anima checkpoint geometry.</summary>
    public static AnimaConfig AnimaPreview3 => new()
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
        ConcatPaddingMask = true,
        ConditionDim = 6144,
        EmbeddedTimestepDim = 2048,
        AdaLnLoraDim = 256,
        RmsNormEps = 1e-6f,
        QkNormEps = 1e-6f,
        LlmAdapter = AnimaLlmAdapterConfig.Default,
    };

    /// <summary>FFN inner dimension after applying <see cref="MlpRatio"/>.</summary>
    public int FfnHiddenSize => (int)(HiddenSize * MlpRatio);

    /// <summary>Total flat patch dim <c>p_t × p_h × p_w</c>.</summary>
    public int PatchVolume => PatchSize.T * PatchSize.H * PatchSize.W;
}

/// <summary>Configuration for the Anima <c>llm_adapter</c> — a 6-block self+cross+MLP transformer that refines
/// Qwen-3 0.6B hidden states before they reach the DiT cross-attention. All values below are measured from the
/// checkpoint.</summary>
public sealed record AnimaLlmAdapterConfig
{
    /// <summary>Adapter hidden dim. <b>Measured: 1024</b> (matches Qwen-3 0.6B hidden).</summary>
    public int HiddenSize { get; init; } = 1024;

    /// <summary>Number of attention heads. <b>Measured: 16</b> (1024 / 64 = 16).</summary>
    public int NumHeads { get; init; } = 16;

    /// <summary>Per-head dim. <b>Measured: 64</b> (q_norm/k_norm scale length).</summary>
    public int HeadDim { get; init; } = 64;

    /// <summary>FFN inner dim. <b>Measured: 4096</b> (= 4 × hidden).</summary>
    public int FfnHiddenSize { get; init; } = 4096;

    /// <summary>Number of stacked adapter blocks. <b>Measured: 6</b>.</summary>
    public int NumLayers { get; init; } = 6;

    /// <summary>Codebook vocab size for the top-level <c>embed.weight</c>. <b>Measured: 32128</b>.
    /// (Equals T5-XXL's vocab — almost certainly an initialization artifact carried over from the original
    /// Cosmos-Predict2 training that the Anima retrain didn't strip. Semantic role TBD; v1 implementation
    /// does not use this codebook in the forward pass.)</summary>
    public int CodebookVocab { get; init; } = 32128;

    /// <summary>RMSNorm epsilon.</summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>QK-norm epsilon.</summary>
    public float QkNormEps { get; init; } = 1e-6f;

    /// <summary>Default = measured Anima config.</summary>
    public static AnimaLlmAdapterConfig Default => new();
}
