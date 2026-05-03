namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for Chroma (Lodestone Rock's 8.9B Flux derivative; <c>lodestones/Chroma</c> on HuggingFace).
///
/// **Important architectural correction (was wrong in earlier scaffolding):** Chroma is **NOT** a "literal Flux fork"
/// despite what the Phase 1 roadmap claimed. It reuses Flux's <c>FluxAttention</c> and <c>FluxPosEmbed</c> primitives,
/// but it replaces Flux's per-block <c>AdaLayerNormZero.linear</c>s with a single shared **distilled-guidance approximator**:
/// a 5-layer SiLU + RMSNorm-residual MLP (<c>ChromaApproximator</c>) that produces a precomputed table of per-block
/// modulation vectors. Each block then **slices** its modulation params from this table instead of computing them
/// from <c>temb</c> via a per-block linear. This eliminates the need for a guidance scalar (<c>guidance_embeds</c>) and
/// CLIP pooled projection — Chroma is conditioned only on T5 + timestep.
///
/// Reference: <c>diffusers/models/transformers/transformer_chroma.py</c> (624 lines), <c>diffusers/pipelines/chroma/pipeline_chroma.py</c>.
/// Also see <c>diffusers/loaders/single_file_utils.py:convert_chroma_transformer_checkpoint_to_diffusers</c> for the
/// BFL→diffusers key remap.
///
/// **Implementation status:** scaffolding only. The transformer + approximator + pruned-AdaLN block variants
/// + checkpoint converter + T5-only pipeline are not yet implemented. See PHASE_4_MODEL_BREADTH.md `### Chroma`
/// for the full task list.
/// </summary>
public record ChromaConfig
{
    /// <summary>Number of dual-stream blocks (`ChromaTransformerBlock`). 19 for the v1 release.</summary>
    public required int Depth { get; init; }

    /// <summary>Number of single-stream blocks (`ChromaSingleTransformerBlock`). 38 for the v1 release.</summary>
    public required int DepthSingleBlocks { get; init; }

    /// <summary>Hidden dim of the transformer (= num_heads * head_dim). 3072 for v1.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of attention heads. 24 for v1.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Per-head dimension. 128 for v1 (24 × 128 = 3072).</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>Approximator input channels. The input is `concat([Timesteps(t,16), Timesteps(0,16), mod_proj(arange,32)])` = 64. Don't change.</summary>
    public int ApproximatorNumChannels { get; init; } = 64;

    /// <summary>Approximator hidden width. 5120 for v1.</summary>
    public int ApproximatorHiddenDim { get; init; } = 5120;

    /// <summary>Number of `PixArtAlphaTextProjection + RMSNorm + residual` layers in the approximator. 5 for v1.</summary>
    public int ApproximatorLayers { get; init; } = 5;

    /// <summary>Approximator output dim = transformer hidden_size = 3072.</summary>
    public int ApproximatorOutDim { get; init; } = 3072;

    /// <summary>Computed: total rows in the per-block modulation table.
    /// <c>3 * num_single_blocks + 12 * num_double_blocks + 2</c>. For 19 doubles + 38 singles → 344 rows.</summary>
    public int ModIndexLength => 3 * DepthSingleBlocks + 12 * Depth + 2;

    /// <summary>Default CFG scale. Chroma is "true CFG" (not distilled-to-1 like Flux Dev). Default 5.0.</summary>
    public float DefaultCfgScale { get; init; } = 5.0f;

    /// <summary>Default inference steps. 35 per the diffusers pipeline default.</summary>
    public int DefaultSteps { get; init; } = 35;

    /// <summary>Chroma v1 preset (`lodestones/Chroma`, ~8.9B params).</summary>
    public static ChromaConfig V1 => new()
    {
        Depth = 19,
        DepthSingleBlocks = 38,
        HiddenSize = 3072,
        NumHeads = 24,
        HeadDim = 128,
    };
}
