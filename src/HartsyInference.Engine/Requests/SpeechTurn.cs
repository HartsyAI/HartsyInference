namespace HartsyInference.Engine.Requests;

/// <summary>A spoken text with its sound: a voice reference (a transcript and its clip) or a turn of a conversation (the text and
/// the codec codes a model generated for it, as <see cref="AudioResult.Turns"/> returns them). A turn carries its sound as
/// <see cref="Codes"/>, as <see cref="Audio"/> the model encodes itself, or both, in which case the codes are used.</summary>
public sealed record SpeechTurn
{
    /// <summary>The transcript of a reference, or the text of a turn.</summary>
    public required string Text { get; init; }

    /// <summary>The clip, for a model to encode into its own codes; null when <see cref="Codes"/> is set.</summary>
    public AudioClip? Audio { get; init; }

    /// <summary>Codec codes, one row per codebook (<c>[codebooks][frames]</c>, the semantic codebook first); null when
    /// <see cref="Audio"/> is set.</summary>
    public IReadOnlyList<int[]>? Codes { get; init; }
}
