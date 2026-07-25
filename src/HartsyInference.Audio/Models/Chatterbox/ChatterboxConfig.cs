using HartsyInference.Audio.Models.LanguageModels.Qwen2;

namespace HartsyInference.Audio.Models.Chatterbox;

/// <summary>Configuration for Resemble AI's Chatterbox TTS. A "T3" Llama-style AR LM turns text tokens into
/// S3 speech tokens (25 Hz, FSQ vocab 6561); the S3Gen stage (a CosyVoice2-derived flow-matching model +
/// HiFTNet vocoder) turns those into 24 kHz audio. Conditioning: a 256-d LSTM voice-encoder embedding + an
/// exaggeration scalar. See <c>docs/Research/CHATTERBOX_ARCHITECTURE.md</c>.
///
/// <para><b>Reuse:</b> S3Gen reuses the existing CosyVoice <c>CosyVoiceFlow</c> + <c>ConditionalCfm</c> +
/// <c>HiFTNetVocoder</c> + <c>S3Tokenizer</c> + <c>CamPlusSpeakerEncoder</c>. Net-new is the T3 LM (a
/// headless <see cref="Qwen2Model"/> + speech/text embeds + learned positions + heads + cond encoder) and the
/// LSTM voice encoder.</para></summary>
public sealed record ChatterboxConfig
{
    /// <summary>T3 backbone — the "Llama_520M" preset (hidden 1024 / 30L / 16 heads MHA / head_dim 64 /
    /// SwiGLU 4096 / RoPE θ=500000 / RMSNorm 1e-5, no bias, untied), with Llama-3 NTK-by-parts RoPE scaling
    /// (factor 8, low 1, high 4, original context 8192) applied through the shared <c>Core.Rope</c> path.</summary>
    public Qwen2Config T3 { get; init; } = new()
    {
        HiddenSize = 1_024,
        NumHiddenLayers = 30,
        NumAttentionHeads = 16,
        NumKeyValueHeads = 16,
        IntermediateSize = 4_096,
        VocabSize = 8_194,           // speech head/embedding size (dummy for the headless backbone)
        MaxPositionEmbeddings = 131_072,
        RopeTheta = 500_000f,
        RopeScaling = new HartsyInference.Core.Rope.RopeScaling
        {
            Type = HartsyInference.Core.Rope.RopeScalingType.Llama3,
            Factor = 8.0,
            LowFreqFactor = 1.0,
            HighFreqFactor = 4.0,
            OriginalContextLength = 8_192,
        },
        RmsNormEps = 1e-5f,
        AttentionBias = false,
        TieWordEmbeddings = false,
    };

    // ── Vocabularies ──
    public int TextVocab { get; init; } = 704;          // English (2454 multilingual)
    public int SpeechVocab { get; init; } = 8_194;      // T3 speech embed/head (S3 codebook 6561 + specials + headroom)
    public int SpeakerEmbedDim { get; init; } = 256;    // LSTM voice-encoder dim
    public int MaxTextTokens { get; init; } = 2_048;
    public int MaxSpeechTokens { get; init; } = 4_096;

    // ── Special tokens ──
    public int StartTextToken { get; init; } = 255;
    public int StopTextToken { get; init; } = 0;
    public int StartSpeechToken { get; init; } = 6_561;
    public int StopSpeechToken { get; init; } = 6_562;

    /// <summary>S3 speech-token rate (Hz) and the underlying FSQ codebook size (3^8).</summary>
    public int SpeechTokenRate { get; init; } = 25;
    public int S3CodebookSize { get; init; } = 6_561;
    public int SampleRate { get; init; } = 24_000;

    // ── Reference-clip conditioning caps (tts.py) ──
    /// <summary>Max cond-prompt speech tokens the T3 perceiver cross-attends (t3 <c>speech_cond_prompt_len</c>).</summary>
    public int SpeechCondPromptLen { get; init; } = 150;
    /// <summary>T3's prompt tokenization sees only the first 6 s of the 16 kHz reference (<c>ENC_COND_LEN</c>).</summary>
    public int T3CondSeconds { get; init; } = 6;
    /// <summary>S3Gen's <c>embed_ref</c> sees only the first 10 s of the reference (<c>DEC_COND_LEN</c>).</summary>
    public int S3GenCondSeconds { get; init; } = 10;

    // ── Generation defaults (tts.py) ──
    public float Temperature { get; init; } = 0.8f;
    public float TopP { get; init; } = 1.0f;
    public float MinP { get; init; } = 0.05f;
    public float RepetitionPenalty { get; init; } = 1.2f;
    public float CfgWeight { get; init; } = 0.5f;
    public float Exaggeration { get; init; } = 0.5f;
    public int MaxNewTokens { get; init; } = 1_000;

    public static ChatterboxConfig Default => new();
}
