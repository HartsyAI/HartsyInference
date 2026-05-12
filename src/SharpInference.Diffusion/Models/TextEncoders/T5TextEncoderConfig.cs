namespace SharpInference.Diffusion.Models.TextEncoders;

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

    /// <summary>If true, each encoder layer has its own learned relative position bias (UMT5 convention,
    /// used by Pile-T5-XL / AuraFlow). If false (T5 v1.1 default), only block 0 has a learned bias and
    /// every subsequent block shares it. Using the wrong mode produces a perfect-looking transformer
    /// with semantically-wrong conditioning — image quality high, prompt ignored.</summary>
    public bool UsePerLayerPositionBias { get; init; } = false;

    /// <summary>T5 v1.1 XXL encoder preset for SD3/Flux text encoding.</summary>
    public static T5TextEncoderConfig Xxl => new()
    {
        DModel = 4096,
        DFf = 10240,
        DKv = 64,
        NumHeads = 64,
        NumLayers = 24,
        VocabSize = 32128,
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
    };
}
