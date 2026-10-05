using System.Text;

namespace HartsyInference.Audio.Models.Auk;

/// <summary>Output-length resolution for AuK: explicit seconds, a reference-text length heuristic, or the reference clip length.</summary>
public static class AukDuration
{
    /// <summary>Latent frames for <paramref name="seconds"/> of audio: <c>max(1, ceil(seconds·sampleRate/hop))</c>.</summary>
    public static int Frames(double seconds, int sampleRate = 24_000, int hop = 480)
    {
        if (!double.IsFinite(seconds) || seconds < 0) throw new ArgumentOutOfRangeException(nameof(seconds));
        return Math.Max(1, (int)Math.Ceiling(seconds * sampleRate / hop));
    }

    /// <summary>Resolves seconds: explicit wins; else <c>refSeconds·bytes(genText)/bytes(refText)/speed</c>; else the reference length; else throws.</summary>
    public static double ResolveSeconds(double? seconds, string? genText, string? refText, double? refSeconds, double speed = 1.0)
    {
        if (!(speed > 0) || !double.IsFinite(speed)) throw new ArgumentOutOfRangeException(nameof(speed), "Speed must be positive.");
        if (seconds is not null)
        {
            if (!(seconds.Value > 0) || !double.IsFinite(seconds.Value)) throw new ArgumentOutOfRangeException(nameof(seconds));
            return seconds.Value;
        }
        if (refSeconds is null)
        {
            throw new ArgumentException("AuK needs an explicit duration when there is no reference audio (instruct TTS).", nameof(seconds));
        }
        if (!(refSeconds.Value > 0) || !double.IsFinite(refSeconds.Value)) throw new ArgumentOutOfRangeException(nameof(refSeconds));
        if (string.IsNullOrEmpty(genText) || string.IsNullOrEmpty(refText)) return refSeconds.Value;
        return refSeconds.Value * Encoding.UTF8.GetByteCount(genText) / Encoding.UTF8.GetByteCount(refText) / speed;
    }

    /// <summary>Convenience: <see cref="ResolveSeconds"/> then <see cref="Frames"/>.</summary>
    public static int ResolveFrames(double? seconds, string? genText, string? refText, double? refSeconds, double speed = 1.0,
        int sampleRate = 24_000, int hop = 480)
        => Frames(ResolveSeconds(seconds, genText, refText, refSeconds, speed), sampleRate, hop);
}
