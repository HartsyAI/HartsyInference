namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>A Drama 3 prompt holds more rows than its limit: <see cref="FishAudioDrama3Config.MaxPromptRows"/>, or less when the
/// loaded model's capacity leaves less room.</summary>
public sealed class FishAudioDrama3PromptTooLongException(int rows, int limit)
    : ArgumentException($"The Drama 3 prompt has {rows} rows; at most {limit} fit.")
{
    /// <summary>Rows the rejected prompt held.</summary>
    public int Rows { get; } = rows;

    /// <summary>Most rows the prompt could hold.</summary>
    public int Limit { get; } = limit;
}
