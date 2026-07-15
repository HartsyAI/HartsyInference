using HartsyInference.Audio.Models.Codecs.Snac;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;

namespace HartsyInference.Audio.Models.Orpheus;

/// <summary>Configuration for Orpheus TTS (<c>canopylabs/orpheus-3b-0.1-ft</c>) — a Llama-3.2-3B causal LM
/// fine-tuned to emit <see cref="SnacConfig">SNAC 24 kHz</see> audio tokens through a single extended-vocab
/// softmax, plus the SNAC codec that turns those tokens back into 24 kHz waveform.
///
/// <para><b>Reuse:</b> the backbone is a plain Llama-3.2-3B → <see cref="Qwen2Model"/> with
/// <see cref="Qwen2Config.AttentionBias"/> off (same path Sesame CSM uses). Sampling reuses
/// <c>NucleusSampler</c> (repetition penalty applied by pre-shaping the logit buffer, the documented usage).
/// Audio decode reuses the built SNAC. The only Orpheus-specific logic is the human-frame prompt wrapping,
/// the 7-token group parse, and the 7→3 hierarchical-layer redistribution.</para>
///
/// <para>The flat audio stream packs <b>7 tokens per SNAC super-frame</b>: position <c>p</c> in
/// <c>[0,6]</c> carries an absolute id <c>AudioCodeBase + p*4096 + code</c>. The 7 positions redistribute
/// into the 3 hierarchical codebooks as 1 (L1) / 2 (L2) / 4 (L3) — matching SNAC 24 kHz strides
/// <c>[4,2,1]</c>.</para></summary>
public sealed record OrpheusConfig
{
    /// <summary>Llama-3.2-3B backbone with the Orpheus extended vocabulary (audio tokens appended).</summary>
    public required Qwen2Config Llm { get; init; }

    /// <summary>SNAC codec that decodes the emitted audio tokens to 24 kHz PCM.</summary>
    public SnacConfig Codec { get; init; } = SnacConfig.Snac24kHz;

    // ── Orpheus special / framing token IDs (canopylabs/orpheus-3b-0.1-ft) ──
    /// <summary>Marks the start of the audio-code span; parse keeps everything after the last occurrence.</summary>
    public int CodeStart { get; init; } = 128_257;
    /// <summary>Start-of-AI-turn token appended to the prompt (before <see cref="CodeStart"/>) to hand the turn
    /// to the model and begin the audio-code stream. Reference <c>orpheus_tts</c> end-tokens are
    /// <c>[EndOfText, EndOfHuman, StartOfAi, CodeStart]</c>.</summary>
    public int StartOfAi { get; init; } = 128_261;
    /// <summary>End-of-speech — the AR stop token.</summary>
    public int EndOfSpeech { get; init; } = 128_258;
    public int StartOfHuman { get; init; } = 128_259;
    public int EndOfHuman { get; init; } = 128_260;
    public int EndOfText { get; init; } = 128_009;
    /// <summary>Base id of the audio-code range; code at group position <c>p</c> = id − (AudioCodeBase + p*4096).</summary>
    public int AudioCodeBase { get; init; } = 128_266;

    /// <summary>SNAC codebook size — the per-position stride between audio-code blocks.</summary>
    public int CodebookSize => Codec.CodebookSize;

    /// <summary>Tokens per SNAC super-frame in the flat stream (1 L1 + 2 L2 + 4 L3).</summary>
    public int TokensPerFrame { get; init; } = 7;

    // ── Sampling defaults (match the reference engine) ──
    public float Temperature { get; init; } = 0.6f;
    public float TopP { get; init; } = 0.95f;
    public int TopK { get; init; } = 0;
    public float RepetitionPenalty { get; init; } = 1.1f;

    /// <summary>Orpheus-3B preset. Backbone is Llama-3.2-3B (hidden 3072 / 28L / 24 heads / GQA 8 KV /
    /// SwiGLU 8192 / RMSNorm 1e-5 / RoPE θ=500000, tied head). Vocab is extended to 156,940 for the audio
    /// tokens. <b>Reconcile on first checkpoint load:</b> the Llama-3.2 "llama3" RoPE NTK-by-parts rescale
    /// (factor 32) is not yet applied — the same deferral as CSM — and the exact extended vocab row count.</summary>
    public static OrpheusConfig Orpheus3B => new()
    {
        Llm = new Qwen2Config
        {
            HiddenSize = 3_072,
            NumHiddenLayers = 28,
            NumAttentionHeads = 24,
            NumKeyValueHeads = 8,
            IntermediateSize = 8_192,
            VocabSize = 156_940,
            MaxPositionEmbeddings = 131_072,
            RopeTheta = 500_000f,
            RmsNormEps = 1e-5f,
            AttentionBias = false,
            TieWordEmbeddings = true,
            BosTokenId = 128_000,
            // EOS used as the AR inference stop (end-of-speech). A config.json-driven converter must NOT
            // overwrite this with 128009 (end-of-text) or 128001 (Llama-3.2 eot); those would break the
            // SNAC super-frame parse which relies on stopping at EndOfSpeech.
            EosTokenId = 128_258,
            PadTokenId = 128_004,
            RopeScaling = new HartsyInference.Core.Rope.RopeScaling
            {
                Type = HartsyInference.Core.Rope.RopeScalingType.Llama3,
                Factor = 32.0,
                OriginalContextLength = 8_192,
                LowFreqFactor = 1.0,
                HighFreqFactor = 4.0,
            },
        },
    };
}
