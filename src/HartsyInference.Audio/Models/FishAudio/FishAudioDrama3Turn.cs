namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>A spoken text and its codec codes: a voice reference (transcript + clip codes) or a finished turn of the
/// conversation (user text + the assistant's generated codes).</summary>
/// <param name="Text">The transcript or the user text of the turn.</param>
/// <param name="Codes">Codec codes <c>[numCodebooks, frames]</c>; row 0 is the semantic codebook.</param>
public sealed record FishAudioDrama3Turn(string Text, int[,] Codes)
{
    /// <summary>Frames the codes span.</summary>
    public int Frames => Codes.GetLength(1);
}
