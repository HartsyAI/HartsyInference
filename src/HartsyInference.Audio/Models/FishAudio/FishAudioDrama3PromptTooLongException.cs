namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>A Drama 3 prompt holds more rows than <see cref="FishAudioDrama3Config.MaxPromptRows"/>.</summary>
public sealed class FishAudioDrama3PromptTooLongException(int rows)
    : ArgumentException($"The Drama 3 prompt has {rows} rows; at most {FishAudioDrama3Config.MaxPromptRows} fit.")
{
    /// <summary>Rows the rejected prompt held.</summary>
    public int Rows { get; } = rows;
}
