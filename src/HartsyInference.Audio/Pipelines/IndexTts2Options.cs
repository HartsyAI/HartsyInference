namespace HartsyInference.Audio.Pipelines;

/// <summary>Generation knobs for <see cref="IndexTts2Pipeline"/>. Covers the GPT's AR-sampling controls
/// (shared shape with <see cref="IndexTtsOptions"/>) plus IndexTTS-2's emotion controls — the real
/// <c>infer_generator</c>'s four emotion modes: none (defaults to the speaker's own voice as the emotion
/// reference, forced <c>alpha=1</c>), an explicit emotion-reference clip, an explicit 8-dim vector, or
/// free-text through <see cref="IndexTts2QwenEmotion"/>.</summary>
/// <remarks>Beam search (<c>num_beams=3</c>, the reference's real default) is not implemented, same documented
/// gap as IndexTTS-1.5 — single-sequence temperature/top-k/top-p decoding only.</remarks>
public sealed record IndexTts2Options
{
    public float Temperature { get; init; } = 0.8f;
    public int TopK { get; init; } = 30;
    public float TopP { get; init; } = 0.8f;
    public float RepetitionPenalty { get; init; } = 10.0f;

    /// <summary>Hard cap on generated mel-code steps; null uses the real reference's own default (1500).</summary>
    public int? MaxMelTokens { get; init; }

    public ulong Seed { get; init; } = 42;

    /// <summary>BCP-47-ish language code (e.g. <c>"en"</c>, <c>"zh"</c>) resolved to the GPT's
    /// <c>lang_embedding</c> row via <see cref="HartsyInference.ModelAssets.Tokenizers.IndexTts2TiktokenTokenizer.LangToToken"/>;
    /// an unrecognized code falls back to the real tokenizer's own <c>"common"</c> row, matching upstream.</summary>
    public string Language { get; init; } = "en";

    /// <summary>Optional separate emotion-reference clip (mono PCM + its sample rate). When null, the main
    /// speaker reference doubles as the emotion reference and <see cref="EmoAlpha"/> is forced to 1.0,
    /// matching the real <c>infer_generator</c>'s own fallback. Ignored when <see cref="EmoVector"/> is set or
    /// <see cref="UseEmoText"/> is true (real source: an explicit vector or text guidance can't be blended via
    /// alpha mixing with an audio reference, so the reference is dropped).</summary>
    public (float[] Audio, int SampleRate)? EmoAudioReference { get; init; }

    /// <summary>Lerp factor between the speaker's own natural emotion and the emotion reference's
    /// (<c>base + alpha*(emo-base)</c>); 1.0 fully adopts the reference's emotion.</summary>
    public float EmoAlpha { get; init; } = 1.0f;

    /// <summary>Explicit 8-dim emotion vector in <see cref="IndexTts2EmotionVectorLookup.EmoNum"/>'s category
    /// order (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm). When set, overrides
    /// <see cref="EmoAudioReference"/> and <see cref="UseEmoText"/>.</summary>
    public float[]? EmoVector { get; init; }

    /// <summary>Real <c>use_random</c>: when an explicit <see cref="EmoVector"/> is given, pick a uniformly
    /// random exemplar per category instead of the cosine-nearest one against the speaker's own style.</summary>
    public bool UseRandomEmoExemplar { get; init; }

    /// <summary>When true, classifies <see cref="EmoText"/> (or, if null, the main synthesis text) through
    /// <see cref="IndexTts2QwenEmotion"/> to derive the emotion vector. Requires the pipeline to have been
    /// loaded with a QwenEmotion classifier — throws otherwise, matching the real source's own
    /// <c>RuntimeError</c> when <c>use_qwen_emo=False</c>.</summary>
    public bool UseEmoText { get; init; }

    /// <summary>Text to classify when <see cref="UseEmoText"/> is set; null uses the main synthesis text.</summary>
    public string? EmoText { get; init; }

    /// <summary>Real <c>duration_factor</c>: scales the length-regulated target frame count
    /// (<c>target_lengths = semanticFrames * 1.72 * durationFactor</c>). 1.0 is the real default.</summary>
    public float DurationFactor { get; init; } = 1.0f;
}
