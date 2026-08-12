namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Configuration for the Gemma 4 decoder transformer used as a conditioning text encoder (LTX-2.5's
/// Gemma-4-12B). Gemma 4 cannot be expressed as a <see cref="LlamaStyleEncoderConfig"/> preset because its
/// geometry varies <i>per layer type</i>: sliding and global layers have different head dimensions, different
/// KV-head counts, different RoPE bases, and the global layers drop <c>v_proj</c> entirely.</summary>
public sealed record Gemma4TextEncoderConfig
{
    /// <summary>Hidden dimension of the residual stream.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Number of transformer blocks.</summary>
    public required int NumLayers { get; init; }

    /// <summary>Number of query heads. Same on sliding and global layers; only the head dimension changes.</summary>
    public required int NumQueryHeads { get; init; }

    /// <summary>Number of key/value heads on <b>sliding</b> layers.</summary>
    public required int NumKvHeads { get; init; }

    /// <summary>Number of key/value heads on <b>global</b> layers (1 for Gemma-4-12B — a single shared KV head).</summary>
    public required int NumGlobalKvHeads { get; init; }

    /// <summary>Per-head dimension on <b>sliding</b> layers.</summary>
    public required int HeadDim { get; init; }

    /// <summary>Per-head dimension on <b>global</b> layers (512 for Gemma-4-12B, double the sliding 256).</summary>
    public required int GlobalHeadDim { get; init; }

    /// <summary>Inner GeGLU MLP dimension.</summary>
    public required int IntermediateSize { get; init; }

    /// <summary>Vocabulary size (input embedding rows). Gemma 4 ties the output head to the embedding, so the
    /// checkpoint carries no <c>lm_head</c> — irrelevant for encoding.</summary>
    public required int VocabSize { get; init; }

    /// <summary>RMSNorm epsilon.</summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>RoPE base for sliding layers.</summary>
    public float SlidingRopeTheta { get; init; } = 10_000f;

    /// <summary>RoPE base for global layers.</summary>
    public float GlobalRopeTheta { get; init; } = 1_000_000f;

    /// <summary>Fraction of the <b>global</b> head dimension that actually rotates; the remaining frequencies are
    /// zeroed (cos 1 / sin 0), leaving those channels untouched. Sliding layers always rotate fully.</summary>
    public float PartialRotaryFactor { get; init; } = 0.25f;

    /// <summary>Number of attended positions (including the current one) on sliding layers.</summary>
    public int SlidingWindow { get; init; } = 1024;

    /// <summary>Length of the repeating layer-type cycle. The <i>last</i> layer of each cycle is a global/full
    /// layer, every other layer is sliding — Gemma-4-12B's 48-entry <c>layer_types</c> list is exactly this
    /// pattern with period 6.</summary>
    public int GlobalLayerPeriod { get; init; } = 6;

    /// <summary>Attention softmax scale. Gemma 4 uses a literal <c>1.0</c>, <b>not</b> <c>1/sqrt(head_dim)</c> —
    /// the scale is folded into the (very large) per-head <c>q_norm</c> weights instead.</summary>
    public float AttentionScale { get; init; } = 1.0f;

    /// <summary>True when global layers reuse the K projection as V (<c>attention_k_eq_v</c>): there is no
    /// <c>v_proj</c> tensor and V is the raw <c>k_proj</c> output taken <b>before</b> <c>k_norm</c> and before RoPE.</summary>
    public bool GlobalAttentionKEqualsV { get; init; } = true;

    /// <summary>Maximum sequence length RoPE tables may be built for; acts as a prompt-length cap.</summary>
    public int MaxPositionEmbeddings { get; init; } = 262_144;

    /// <summary>Padding token id.</summary>
    public int PadTokenId { get; init; } = 0;

    /// <summary>End-of-sequence token id.</summary>
    public int EosTokenId { get; init; } = 1;

    /// <summary>Beginning-of-sequence token id (the encoder does not insert it; the tokenizer does).</summary>
    public int BosTokenId { get; init; } = 2;

    /// <summary>Gemma-3n per-layer-input width. Gemma-4-12B has this mechanism OFF; a nonzero value is rejected
    /// rather than silently ignored.</summary>
    public int HiddenSizePerLayerInput { get; init; } = 0;

    /// <summary>Number of trailing layers that reuse an earlier layer's KV. Gemma-4-12B has none; a nonzero value
    /// is rejected rather than silently ignored.</summary>
    public int NumKvSharedLayers { get; init; } = 0;

    /// <summary>Whether the last of the <c>NumLayers + 1</c> hidden states returned by
    /// <see cref="Gemma4TextEncoder.EncodeMultiLayer"/> passes through <c>model.norm</c>. True reproduces the
    /// reference (states <c>0 … NumLayers-1</c> raw, the final state normed); false returns every state raw,
    /// which is what the existing <see cref="LlamaStyleEncoder"/> Gemma-3 path does.</summary>
    public bool ApplyFinalNormToLastState { get; init; } = true;

    /// <summary>Multiplier applied to token embeddings right after lookup (the Gemma "normalizer").</summary>
    public float EmbeddingScale => (float)Math.Sqrt(HiddenSize);

    /// <summary>Q projection output width on the given layer.</summary>
    public int QDimFor(int layerIndex) => NumQueryHeads * HeadDimFor(layerIndex);

    /// <summary>K/V projection output width on the given layer.</summary>
    public int KvDimFor(int layerIndex) => KvHeadsFor(layerIndex) * HeadDimFor(layerIndex);

    /// <summary>True when the layer is a global/full-attention layer (the last of each <see cref="GlobalLayerPeriod"/> cycle).</summary>
    public bool IsGlobalLayer(int layerIndex) => layerIndex % GlobalLayerPeriod == GlobalLayerPeriod - 1;

    /// <summary>Per-head dimension for the given layer.</summary>
    public int HeadDimFor(int layerIndex) => IsGlobalLayer(layerIndex) ? GlobalHeadDim : HeadDim;

    /// <summary>Key/value head count for the given layer.</summary>
    public int KvHeadsFor(int layerIndex) => IsGlobalLayer(layerIndex) ? NumGlobalKvHeads : NumKvHeads;

    /// <summary>Whether the given layer omits <c>v_proj</c> and reuses the raw K projection as V.</summary>
    public bool KEqualsVFor(int layerIndex) => GlobalAttentionKEqualsV && IsGlobalLayer(layerIndex);

    /// <summary>Builds the RoPE inverse frequencies for one layer type: <c>headDim/2</c> entries, of which the
    /// leading <c>rotaryPairs</c> are real and the remainder are zero (identity rotation on those channels).
    /// Global layers use <c>int(partial_rotary_factor · globalHeadDim / 2)</c> real pairs with the exponent still
    /// divided by the <b>full</b> head dimension; sliding layers rotate every pair.</summary>
    public double[] BuildInverseFrequencies(int layerIndex)
    {
        int headDim = HeadDimFor(layerIndex);
        int half = headDim / 2;
        bool global = IsGlobalLayer(layerIndex);
        int rotaryPairs = global ? (int)Math.Floor((double)PartialRotaryFactor * headDim / 2.0) : half;
        if (rotaryPairs < 0 || rotaryPairs > half)
            throw new InvalidOperationException($"Partial rotary factor {PartialRotaryFactor} yields {rotaryPairs} pairs for head dim {headDim}.");
        double theta = global ? GlobalRopeTheta : SlidingRopeTheta;
        double[] inv = new double[half];
        for (int k = 0; k < rotaryPairs; k++)
            inv[k] = 1.0 / Math.Pow(theta, (double)(2 * k) / headDim);
        return inv;
    }

    /// <summary>LTX-2.5's text encoder: Gemma-4-12B (48 layers, 3840 hidden), verified against the checkpoint's
    /// own <c>gemma_config.text_config</c> metadata.</summary>
    public static Gemma4TextEncoderConfig Gemma4_12B { get; } = new()
    {
        HiddenSize = 3840,
        NumLayers = 48,
        NumQueryHeads = 16,
        NumKvHeads = 8,
        NumGlobalKvHeads = 1,
        HeadDim = 256,
        GlobalHeadDim = 512,
        IntermediateSize = 15360,
        VocabSize = 262144,
        RmsNormEps = 1e-6f,
        SlidingRopeTheta = 10_000f,
        GlobalRopeTheta = 1_000_000f,
        PartialRotaryFactor = 0.25f,
        SlidingWindow = 1024,
        GlobalLayerPeriod = 6,
    };
}
