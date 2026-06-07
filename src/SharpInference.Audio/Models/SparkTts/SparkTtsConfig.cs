using SharpInference.Audio.Models.LanguageModels.Qwen2;

namespace SharpInference.Audio.Models.SparkTts;

/// <summary>Configuration for Spark-TTS-0.5B (`SparkAudio/Spark-TTS-0.5B`). Spark-TTS is an LLM-based
/// zero-shot voice-cloning TTS: a fine-tuned Qwen2.5-0.5B causal LM (vocab extended 151,936 → 166,000
/// with BiCodec audio tokens + control tokens) autoregressively emits 32 global tokens then a stream of
/// 50 Hz semantic tokens, which BiCodec decodes to a 16 kHz waveform. See
/// <c>docs/Research/SPARK_TTS_ARCHITECTURE.md</c>.
///
/// <para><b>Reuse:</b> the LM is a plain Qwen2.5-0.5B (single extended-vocab softmax through the standard
/// tied <c>lm_head</c>) — reuses <see cref="Qwen2Model"/> directly, unlike CosyVoice which has separate
/// speech heads. BiCodec decode reuses FSQ (global), VQ embedding (semantic), and DAC/Vocos-style
/// Snake + ConvTranspose wave-generator patterns.</para>
///
/// <para><b>Token-offset reconciliation pending:</b> the exact audio/control token IDs come from the
/// checkpoint's <c>added_tokens.json</c>. The offsets below are the documented layout; they're config
/// fields so reconciliation against the real file is a one-place edit. The LM emits absolute token IDs;
/// the pipeline maps them back to codec indices via these bases.</para></summary>
public sealed record SparkTtsConfig
{
    /// <summary>Qwen2.5-0.5B backbone with the Spark-extended vocab (166,000). Single softmax over the
    /// tied <c>lm_head</c> — audio + control tokens are real vocab entries.</summary>
    public required Qwen2Config Llm { get; init; }

    /// <summary>BiCodec semantic VQ codebook size (50 Hz single stream).</summary>
    public int SemanticVocab { get; init; } = 8_192;

    /// <summary>BiCodec global FSQ codebook size (fixed 32-token speaker set).</summary>
    public int GlobalVocab { get; init; } = 4_096;

    /// <summary>Number of global tokens emitted per utterance.</summary>
    public int NumGlobalTokens { get; init; } = 32;

    /// <summary>Output sample rate.</summary>
    public int SampleRate { get; init; } = 16_000;

    // ── Token-ID offsets into the LM vocab (reconcile against added_tokens.json) ──
    public int SemanticTokenBase { get; init; } = 151_661;            // <|bicodec_semantic_0|>
    public int GlobalTokenBase { get; init; } = 151_661 + 8_192;      // <|bicodec_global_0|>
    public int EosTokenId { get; init; } = 151_645;                   // standard Qwen2.5 EOS
    public int StartGlobalTokenId { get; init; } = 151_653;
    public int EndGlobalTokenId { get; init; } = 151_654;
    public int StartSemanticTokenId { get; init; } = 151_655;

    /// <summary>BiCodec wave-generator (semantic+global tokens → 16 kHz waveform) config.</summary>
    public SparkBiCodecConfig BiCodec { get; init; } = new();

    /// <summary>LM autoregressive sampling defaults.</summary>
    public float Temperature { get; init; } = 0.8f;
    public int TopK { get; init; } = 50;
    public float TopP { get; init; } = 0.95f;

    /// <summary>`SparkAudio/Spark-TTS-0.5B` preset.</summary>
    public static SparkTtsConfig V0_5B => new()
    {
        Llm = Qwen2Config.Qwen25_0_5B with { VocabSize = 166_000 },
    };
}

/// <summary>BiCodec decode-side config — a DAC-style HiFi-GAN wave generator (Snake1d + transposed convs)
/// fed by Vocos/ConvNeXt prenet conditioned on a global speaker d-vector. Reuses the same op families as
/// the DAC decoder + Vocos.</summary>
public sealed record SparkBiCodecConfig
{
    /// <summary>Semantic VQ embedding dim (factorized codebook projected up).</summary>
    public int SemanticDim { get; init; } = 1_024;

    /// <summary>Global FSQ levels — 6 channels × 4 levels = 4096 codes.</summary>
    public int[] FsqLevels { get; init; } = [4, 4, 4, 4, 4, 4];

    /// <summary>Global speaker d-vector dim (FSQ → linear → 1024).</summary>
    public int GlobalDim { get; init; } = 1_024;

    /// <summary>Wave-generator base channels.</summary>
    public int UpsampleInitialChannel { get; init; } = 1_024;

    /// <summary>Transposed-conv upsample rates [8, 5, 4, 2] (product 320 = hop).</summary>
    public int[] UpsampleRates { get; init; } = [8, 5, 4, 2];
    public int[] UpsampleKernelSizes { get; init; } = [16, 11, 8, 4];
    public int[] ResBlockKernelSizes { get; init; } = [3, 7, 11];
    public int[][] ResBlockDilationSizes { get; init; } = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];

    /// <summary>Prenet ConvNeXt block count (Vocos backbone conditioned on the d-vector).</summary>
    public int PrenetBlocks { get; init; } = 12;
    public int MelBins { get; init; } = 80;
}
