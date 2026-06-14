namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for AuraFlow MMDiT transformer (fal/AuraFlow-v0.3, ~6.8B params).
/// Architecture: 4 dual-stream <c>AuraFlowJointTransformerBlock</c>s followed by 32 single-stream
/// <c>AuraFlowSingleTransformerBlock</c>s. Text encoder is **Pile-T5-XL** (output dim 2048,
/// not the 4096 of T5-XXL); VAE is the SDXL 4-channel KL.
///
/// Key architectural quirks (vs Flux/SD3):
/// - **FP32 LayerNorm everywhere** (`fp32_layer_norm` for `AdaLayerNormZero` + QK-norm + final).
/// - **No biases on attention or FFN linears** (`bias=False`, `out_bias=False`).
/// - **SwiGLU FFN** with `mlp_dim = find_multiple(int(2*4*dim/3), 256)` = **8192** for v0.3 (NOT 4×hidden).
/// - **8 register tokens** prepended to text after `context_embedder` (anti-attn-artifact trick from <https://huggingface.co/papers/2309.16588>).
/// - **Patch embed is a `nn.Linear`** over flattened patches (not Conv2d). Uses learned `pos_embed.pos_embed` of shape `[1, pos_embed_max_size, embed_dim]` selected via `pe_selection_index_based_on_dim` against a √1024=32×32 grid.
/// - **`caption_projection_dim` = `inner_dim`** (3072 = 12*256) in the released v0.3 checkpoint.
/// - **`joint_attention_dim` = 2048** matches Pile-T5-XL hidden_size.
/// </summary>
public sealed record AuraFlowConfig
{
    /// <summary>Hidden dimension of the transformer (= num_attention_heads * attention_head_dim).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension (256 for v0.3).</summary>
    public int HeadDim { get; init; } = 256;

    /// <summary>Number of joint dual-stream MMDiT blocks (image + text streams with separate AdaLN).</summary>
    public required int NumDoubleBlocks { get; init; }

    /// <summary>Number of single-stream blocks (operating on concat([text, image]) tokens).</summary>
    public required int NumSingleBlocks { get; init; }

    /// <summary>Patch size for the patch embed Linear projection.</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Number of latent channels (4 for SDXL VAE).</summary>
    public int InChannels { get; init; } = 4;

    /// <summary>Output channels (typically equal to <see cref="InChannels"/>).</summary>
    public int OutChannels { get; init; } = 4;

    /// <summary>Pile-T5-XL output dimension. v0.3 = 2048.</summary>
    public int ContextDim { get; init; } = 2048;

    /// <summary>Caption projection dim — projects context to <see cref="HiddenSize"/>. v0.3 = 3072 = inner_dim.</summary>
    public int CaptionProjectionDim { get; init; } = 3072;

    /// <summary>Maximum positional-embedding grid size (`pos_embed_max_size`). 1024 = 32×32 grid for v0.3.</summary>
    public int PosEmbedMaxSize { get; init; } = 1024;

    /// <summary>Number of register tokens prepended to text after `context_embedder`. v0.3 = 8.</summary>
    public int NumRegisterTokens { get; init; } = 8;

    /// <summary>SwiGLU FFN inner dim. <c>find_multiple(int(2*4*dim/3), 256)</c> → 8192 for dim=3072.</summary>
    public int MlpDim { get; init; } = 8192;

    /// <summary>QK-norm epsilon. AuraFlow uses <c>FP32LayerNorm</c> with default eps.</summary>
    public float QkNormEps { get; init; } = 1e-5f;

    /// <summary>Whether QK-norm is applied. AuraFlow always uses it (`fp32_layer_norm`).</summary>
    public bool UseQkNorm { get; init; } = true;

    /// <summary>fal/AuraFlow-v0.3 preset (~6.8B params). Defaults match diffusers
    /// <c>AuraFlowTransformer2DModel.__init__</c> (transformer_qwenimage.py:303-317):
    /// 12 heads × 256 head_dim = 3072 inner_dim, 4 + 32 blocks, joint_dim=2048, pos_embed_max_size=1024, 8 register tokens.</summary>
    public static AuraFlowConfig V03 => new()
    {
        HiddenSize = 3072,
        NumHeads = 12,
        HeadDim = 256,
        NumDoubleBlocks = 4,
        NumSingleBlocks = 32,
        PatchSize = 2,
        InChannels = 4,
        OutChannels = 4,
        ContextDim = 2048,
        CaptionProjectionDim = 3072,
        PosEmbedMaxSize = 1024,
        NumRegisterTokens = 8,
        MlpDim = 8192,
        UseQkNorm = true,
    };
}
