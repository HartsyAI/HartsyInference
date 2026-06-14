namespace HartsyInference.Audio.Pipelines;

/// <summary>Knobs for a single transcription call. All fields have safe defaults
/// matching OpenAI Whisper's <c>transcribe()</c> behavior for English / multilingual
/// audio. The pipeline does NOT cache options between calls — pass a fresh instance
/// per audio chunk.</summary>
public sealed record WhisperOptions
{
    /// <summary>Language code (e.g., <c>"en"</c>, <c>"zh"</c>) or <c>null</c> to skip the
    /// language token entirely (English-only models). Set to <c>"auto"</c> to request
    /// automatic detection — not yet implemented; throws.</summary>
    public string? Language { get; init; } = "en";

    /// <summary>If true, emit translation (to English) instead of transcription. Most
    /// models support this; check the model card before relying on it.</summary>
    public bool Translate { get; init; }

    /// <summary>Whether to emit timestamp tokens. When false, <c>&lt;|notimestamps|&gt;</c>
    /// is prepended and only text tokens appear in the output.</summary>
    public bool WithTimestamps { get; init; }

    /// <summary>Maximum new tokens to generate before forcing EOT. The decoder will
    /// also stop on natural EOT. Default 224 matches OpenAI's behavior (half of
    /// <c>n_text_ctx=448</c>, leaving room for the prompt prefix).</summary>
    public int MaxNewTokens { get; init; } = 224;

    /// <summary>Greedy sampling only in v1; temperature fallback and beam search land
    /// later. This field exists so callers can opt into future behavior.</summary>
    public float Temperature { get; init; } = 0f;
}
