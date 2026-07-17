namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Configuration for the T5 v1.1 XXL encoder-only model used by SD3 and Flux.</summary>
public sealed record T5TextEncoderConfig
{
    /// <summary>Hidden dimension (embedding size).</summary>
    public required int DModel { get; init; }

    /// <summary>Feed-forward intermediate dimension.</summary>
    public required int DFf { get; init; }

    /// <summary>Per-head key/value dimension.</summary>
    public required int DKv { get; init; }

    /// <summary>Number of attention heads.</summary>
    public required int NumHeads { get; init; }

    /// <summary>Number of encoder layers.</summary>
    public required int NumLayers { get; init; }

    /// <summary>Vocabulary size.</summary>
    public required int VocabSize { get; init; }

    /// <summary>Number of buckets for relative position bias.</summary>
    public int RelativeAttentionNumBuckets { get; init; } = 32;

    /// <summary>Maximum distance for relative position bucketing. Beyond this, all positions share one bucket.</summary>
    public int RelativeAttentionMaxDistance { get; init; } = 128;

    /// <summary>RMSNorm epsilon.</summary>
    public float LayerNormEpsilon { get; init; } = 1e-6f;

    /// <summary>If true (T5 v1.1 / UMT5 GEGLU: <c>wi_0</c>/<c>wi_1</c> pair), the feed-forward is gated.
    /// If false (original T5 v1.0, e.g. <c>google/t5-base</c>), it is a plain two-layer MLP with a single
    /// <c>wi</c> projection.</summary>
    public bool GatedFeedForward { get; init; } = true;

    /// <summary>If true, the feed-forward activation is ReLU (original T5 v1.0 <c>DenseReluDense</c>);
    /// if false (v1.1 default), GELU.</summary>
    public bool UseReluActivation { get; init; } = false;

    /// <summary>Attention score scale. T5 (all versions, per the Mesh-TensorFlow "avoid scaling before softmax"
    /// design) uses <b>1.0</b> — the query is NOT divided by <c>sqrt(head_dim)</c>; the relative position bias and
    /// initialization compensate. Set this to <c>1.0f</c> for a faithful T5 (MusicGen's <c>google/t5-base</c> needs
    /// it). Left <c>null</c> the block falls back to <c>1/sqrt(head_dim)</c> (the prior engine default kept for the
    /// already-shipping Flux/SD3 T5-XXL path until that path is bit-exactly re-validated).</summary>
    public float? AttentionScale { get; init; }

    /// <summary>If true, each encoder layer has its own learned relative position bias (UMT5 convention,
    /// used by Pile-T5-XL / AuraFlow). If false (T5 v1.1 default), only block 0 has a learned bias and
    /// every subsequent block shares it. Using the wrong mode produces a perfect-looking transformer
    /// with semantically-wrong conditioning — image quality high, prompt ignored.</summary>
    public bool UsePerLayerPositionBias { get; init; } = false;

    /// <summary>T5 v1.1 XXL encoder preset for SD3/Flux/LTX text encoding.</summary>
    public static T5TextEncoderConfig Xxl => new()
    {
        DModel = 4096,
        DFf = 10240,
        DKv = 64,
        NumHeads = 64,
        NumLayers = 24,
        VocabSize = 32128,
        AttentionScale = 1.0f,   // faithful T5 (all versions): no 1/sqrt(head_dim) scaling. The legacy null
                                 // fallback (1/sqrt(dkv)) detuned every layer's softmax — same class of bug
                                 // that produced Wan's flat videos via the Umt5Xxl preset. NOTE: this changes
                                 // Flux/SD3/LTX conditioning (now matches the HF/Comfy reference); Flux/SD3
                                 // outputs will shift and should be spot-checked.
    };

    /// <summary>Original T5 v1.0 base encoder preset (`google/t5-base`) — the text encoder MusicGen / AudioGen
    /// condition on. Unlike the v1.1 / UMT5 presets above, v1.0 uses a NON-gated feed-forward (single <c>wi</c>
    /// projection) with ReLU activation, and shares block 0's relative position bias across all layers. Standard
    /// 32k T5 SentencePiece vocab (Google's <c>t5_spiece.model</c>, not the 256k multilingual one).</summary>
    public static T5TextEncoderConfig T5Base => new()
    {
        DModel = 768,
        DFf = 3072,
        DKv = 64,
        NumHeads = 12,
        NumLayers = 12,
        VocabSize = 32128,
        GatedFeedForward = false,
        UseReluActivation = true,
        AttentionScale = 1.0f,   // faithful T5: no 1/sqrt(head_dim) scaling.
        // UsePerLayerPositionBias stays false: T5 v1.0 learns the bias in block 0 and shares it.
    };

    /// <summary>t5-large encoder preset (`google-t5/t5-large`) — AudioGen's conditioning encoder (NOT t5-base;
    /// AudioGen's <c>enc_to_dec_proj</c> input is 1024). 24 layers, hidden 1024, 16 heads. Same T5 v1.0 family
    /// as <see cref="T5Base"/> (shared block-0 position bias, ReLU FFN, no attention scaling).</summary>
    public static T5TextEncoderConfig T5Large => new()
    {
        DModel = 1024,
        DFf = 4096,
        DKv = 64,
        NumHeads = 16,
        NumLayers = 24,
        VocabSize = 32128,
        GatedFeedForward = false,
        UseReluActivation = true,
        AttentionScale = 1.0f,
    };

    /// <summary>Pile-T5-XL (UMT5) encoder preset for AuraFlow text encoding (`EleutherAI/pile-t5-xl`).
    /// <c>d_model = 2048</c> matches AuraFlow's <c>joint_attention_dim = 2048</c>. UMT5 differs from
    /// T5 v1.1 in TWO ways for the encoder path: (1) the SentencePiece vocabulary (different token IDs
    /// for the same words — needs the Pile-T5 <c>spiece.model</c>, not Google's <c>t5_spiece.model</c>),
    /// AND (2) a separately-learned <c>relative_attention_bias</c> per layer rather than a single
    /// shared one. Both are required — using the wrong tokenizer or sharing block 0's bias across
    /// all 24 layers produces a perfect-looking transformer with semantically wrong conditioning
    /// (image high quality, prompt ignored). Max sequence length 256 per AuraFlow's pipeline.</summary>
    public static T5TextEncoderConfig PileT5Xl => new()
    {
        DModel = 2048,
        DFf = 5120,
        DKv = 64,
        NumHeads = 32,
        NumLayers = 24,
        VocabSize = 32128,
        UsePerLayerPositionBias = true,
        AttentionScale = 1.0f,   // faithful T5/Pile-T5: no 1/sqrt(head_dim) scaling (matches T5Base).
    };

    /// <summary>T5-11B encoder preset (`google-t5/t5-11b`) — the text encoder Cosmos-Predict1 Video2World conditions
    /// its per-layer cross-attention on (<c>context_dim = 1024 = d_model</c>). NOT T5-XXL (which is 4096-dim); T5-11B is
    /// the original T5 v1.0 family: <c>d_model=1024, d_ff=65536, d_kv=128, num_heads=64, num_layers=24</c>, non-gated
    /// ReLU feed-forward, block-0-shared relative position bias, standard 32k T5 SentencePiece vocab. Cosmos encodes at
    /// <c>max_length=512</c> and zero-masks positions past the real prompt length. The wide <c>d_kv=128</c> (inner
    /// attention dim <c>64·128 = 8192</c>, distinct from <c>d_model=1024</c>) is the T5-11B signature.</summary>
    public static T5TextEncoderConfig T5_11B => new()
    {
        DModel = 1024,
        DFf = 65536,
        DKv = 128,
        NumHeads = 64,
        NumLayers = 24,
        VocabSize = 32128,
        GatedFeedForward = false,
        UseReluActivation = true,
        AttentionScale = 1.0f,   // faithful T5 v1.0: no 1/sqrt(head_dim) scaling. UsePerLayerPositionBias stays false
                                 // (v1.0 learns the bias in block 0 and shares it).
    };

    /// <summary>umT5-base encoder preset (`google/umt5-base`) used by ACE-Step's style/genre conditioning — 12 layers,
    /// hidden 768, per-layer relative position bias, the 256k multilingual SentencePiece (same family as
    /// <see cref="Umt5Xxl"/>, just the base size).</summary>
    public static T5TextEncoderConfig Umt5Base => new()
    {
        DModel = 768,
        DFf = 2048,
        DKv = 64,
        NumHeads = 12,
        NumLayers = 12,
        VocabSize = 256384,
        UsePerLayerPositionBias = true,
    };

    /// <summary>umT5-XXL encoder preset (`google/umt5-xxl`) used by Wan-Video text conditioning. Same dims as T5-XXL
    /// but with the UMT5 conventions: per-layer learned relative position bias AND the 256k multilingual
    /// SentencePiece vocabulary (needs the umT5 <c>spiece.model</c>, not Google's T5 one). Max sequence length 512
    /// per Wan's pipeline.</summary>
    public static T5TextEncoderConfig Umt5Xxl => new()
    {
        DModel = 4096,
        DFf = 10240,
        DKv = 64,
        NumHeads = 64,
        NumLayers = 24,
        VocabSize = 256384,
        UsePerLayerPositionBias = true,
        AttentionScale = 1.0f,   // faithful T5/UMT5: no 1/sqrt(head_dim) scaling. Leaving this null (legacy
                                 // 1/sqrt(dkv) fallback) detunes every encoder layer's softmax and produces
                                 // semantically-garbage Wan conditioning — the flat/uniform-color video bug.
    };
}
