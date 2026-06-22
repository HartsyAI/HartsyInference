using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Audio.Models.LanguageModels.Qwen3;

namespace HartsyInference.Audio.Models.QwenTts;

/// <summary>Top-level configuration for Qwen3-TTS (12 Hz). Bundles the talker backbone (<see cref="Qwen3Config"/>
/// + dual embedding sizes + codec head), the MTP code-predictor (5-layer depth transformer + 15 codebooks), the
/// Mimi codec used for reference-audio encoding (clone mode), the custom codec decoder, the codec control ids,
/// and the 9 built-in custom-voice speaker ids. See <c>docs/Research/QWEN3_TTS_ARCHITECTURE.md</c>.</summary>
public sealed record Qwen3TtsConfig
{
    // ── Talker ──────────────────────────────────────────────────────────
    /// <summary>Qwen3 backbone for the talker (1.7B preset by default).</summary>
    public required Qwen3Config Talker { get; init; }

    /// <summary>Text embedding vocab (Qwen tokenizer).</summary>
    public int TextVocabSize { get; init; } = 151_936;

    /// <summary>Codec embedding vocab fed back into the talker (codebook 0 + control tokens).</summary>
    public int CodecVocabSize { get; init; } = 3_072;

    /// <summary>Output width of <c>codec_head</c> (predicts codebook 0). Equals <see cref="CodecVocabSize"/>.</summary>
    public int CodecHeadOut { get; init; } = 3_072;

    /// <summary>Number of code groups (RVQ codebooks total across talker + MTP).</summary>
    public int NumCodeGroups { get; init; } = 16;

    /// <summary>3D mRoPE band split (must sum to <c>Talker.HeadDim / 2</c>; default <c>[24,20,20]</c> for head_dim 128).</summary>
    public int[] MropeSection { get; init; } = [24, 20, 20];

    // ── Code predictor (MTP) ────────────────────────────────────────────
    /// <summary>Qwen3 backbone for the 5-layer MTP depth transformer (plain 1D RoPE).</summary>
    public required Qwen3Config CodePredictor { get; init; }

    /// <summary>Codebooks predicted by the MTP (1..15) — 15 separate embeddings + 15 heads.</summary>
    public int MtpCodebooks { get; init; } = 15;

    /// <summary>Per-codebook vocabulary inside the MTP (acoustic codebook size).</summary>
    public int MtpVocabSize { get; init; } = 2_048;

    // ── Codec ───────────────────────────────────────────────────────────
    /// <summary>Mimi config (the codec ENCODER for ref-audio in clone mode; first 16 of 32 quantizers).</summary>
    public required MimiConfig Codec { get; init; }

    /// <summary>Custom codec-decoder config (net-new Snake/ConvNeXt causal vocoder).</summary>
    public required Qwen3TtsVocoderConfig Vocoder { get; init; }

    // ── Codec control ids (1.7B) ────────────────────────────────────────
    public int CodecPad { get; init; } = 2_148;
    public int CodecBos { get; init; } = 2_149;
    public int CodecEos { get; init; } = 2_150;
    public int CodecThink { get; init; } = 2_154;
    public int CodecNoThink { get; init; } = 2_155;
    public int CodecThinkBos { get; init; } = 2_156;
    public int CodecThinkEos { get; init; } = 2_157;

    /// <summary>First language-id codec token. Language ids span 2050..2074.</summary>
    public int LanguageIdBase { get; init; } = 2_050;
    public int LanguageIdCount { get; init; } = 25;

    /// <summary>Real codec codebook-0 entries are 0..2047; everything &gt;= this is a control token.</summary>
    public int CodecRealVocab { get; init; } = 2_048;

    // ── Text special ids ────────────────────────────────────────────────
    public int TextImStart { get; init; } = 151_644;
    public int TextImEnd { get; init; } = 151_645;
    public int TextAssistant { get; init; } = 77_091;
    public int TextAudioStart { get; init; } = 151_669;
    public int TextAudioEnd { get; init; } = 151_670;
    public int TextTtsPad { get; init; } = 151_671;
    public int TextTtsBos { get; init; } = 151_672;
    public int TextTtsEod { get; init; } = 151_673;

    // ── Modes / sampling ────────────────────────────────────────────────
    /// <summary>The 9 built-in custom-voice speaker ids (codec-space tokens prefixed before generation).</summary>
    public IReadOnlyList<int> CustomVoiceSpeakerIds { get; init; } = [2_075, 2_076, 2_077, 2_078, 2_079, 2_080, 2_081, 2_082, 2_083];

    public float Temperature { get; init; } = 0.9f;
    public int TopK { get; init; } = 50;
    public float TopP { get; init; } = 1.0f;
    public float RepetitionPenalty { get; init; } = 1.05f;
    public int MinNewTokens { get; init; } = 2;
    public int MaxNewTokens { get; init; } = 2_048;

    /// <summary>Auto codec prefill written before the talker starts emitting: [nothink, think_bos, think_eos, pad, bos].</summary>
    public int[] CodecPrefill() => [CodecNoThink, CodecThinkBos, CodecThinkEos, CodecPad, CodecBos];

    /// <summary>Qwen3-TTS-12Hz-1.7B preset.</summary>
    public static Qwen3TtsConfig Default_1_7B => new()
    {
        Talker = Qwen3Config.Talker1_7B,
        CodePredictor = Qwen3Config.CodePredictor,
        Codec = MimiConfig.Mimi24kHz with { AcousticCodebooks = 15 },
        Vocoder = Qwen3TtsVocoderConfig.Default,
    };
}
