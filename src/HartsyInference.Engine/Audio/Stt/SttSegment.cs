namespace HartsyInference.Engine.Audio;

/// <summary>One timestamped span of a transcript: a single word when <see cref="IsWord"/> (Whisper, aligned through cross-attention), otherwise whatever the model natively emits — for Whisper's <c>&lt;|t|&gt;</c> tokens a phrase/utterance chunk.</summary>
internal sealed record SttSegment
{
    /// <summary>The span's transcript text. A word keeps the leading space that marks it as following another word.</summary>
    internal required string Text { get; init; }

    /// <summary>Start time in seconds.</summary>
    internal required double Start { get; init; }

    /// <summary>End time in seconds.</summary>
    internal required double End { get; init; }

    /// <summary>Whether this span is one word, so its text joins its neighbours' without a separator.</summary>
    internal bool IsWord { get; init; }

    /// <summary>The transcript the spans spell: words concatenate as they are, phrases are trimmed and joined by a space.</summary>
    internal static string Join(IReadOnlyList<SttSegment> spans)
    {
        if (spans.Count > 0 && spans[0].IsWord)
        {
            return string.Concat(spans.Select(span => span.Text)).Trim();
        }
        return string.Join(" ", spans.Select(span => span.Text.Trim())).Trim();
    }
}
