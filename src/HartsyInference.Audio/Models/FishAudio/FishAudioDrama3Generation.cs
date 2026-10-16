namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>One Drama 3 generation: the codes it produced and why it stopped.</summary>
/// <param name="Codes">Codec codes <c>[numCodebooks, frames]</c>, without the eos frame.</param>
/// <param name="ReachedEos">True when the model ended the turn; false when the frame budget ran out first.</param>
public sealed record FishAudioDrama3Generation(int[,] Codes, bool ReachedEos)
{
    /// <summary>Frames generated.</summary>
    public int Frames => Codes.GetLength(1);
}
