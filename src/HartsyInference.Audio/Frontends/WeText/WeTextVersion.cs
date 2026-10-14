namespace HartsyInference.Audio.Frontends.WeText;

/// <summary>The wetext release whose <c>normalize</c> a <see cref="WeTextNormalizer"/> reproduces. Both run the same
/// grammars; they differ in when the grammars run and in how the token fields are re-serialized between tagger and
/// verbalizer.</summary>
public enum WeTextVersion
{
    /// <summary>wetext 0.0.4 (CosyVoice 2's pin): only text with a decimal digit is normalized, and the reorder writes
    /// field values raw and drops fields outside the order table.</summary>
    V0_0_4,

    /// <summary>wetext 0.1.8 (what IndexTTS's <c>wetext&gt;=0.0.9</c> resolves to): text is stripped, English is always
    /// normalized and Chinese only with a decimal digit, and the reorder re-escapes values and keeps extra fields.</summary>
    V0_1_8,
}
