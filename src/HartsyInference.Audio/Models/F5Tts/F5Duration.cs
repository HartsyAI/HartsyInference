namespace HartsyInference.Audio.Models.F5Tts;

/// <summary>F5-TTS output-length heuristic; AuK's zero-shot duration reuses the same rule.</summary>
public static class F5Duration
{
    /// <summary>Generations under this many UTF-8 bytes use <see cref="ShortTextSpeed"/>.</summary>
    public const int ShortTextByteThreshold = 10;

    /// <summary>Speed divisor for short generations; the reference ratio alone gives them too few frames.</summary>
    public const double ShortTextSpeed = 0.3;

    /// <summary>Speed divisor for a generation of <paramref name="genBytes"/> UTF-8 bytes.</summary>
    public static double LocalSpeed(int genBytes, double speed) => genBytes < ShortTextByteThreshold ? ShortTextSpeed : speed;

    /// <summary>Target mel frames: reference frames scaled by the UTF-8 length ratio, truncated; an empty reference divides by one.</summary>
    public static int TargetFrames(int refFrames, int refBytes, int genBytes, double speed)
        => (int)(refFrames / (double)Math.Max(1, refBytes) * genBytes / LocalSpeed(genBytes, speed));
}
