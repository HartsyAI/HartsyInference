using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>Decides which waveform ships with a generated video, so every family gets the same behaviour instead of each pipeline inventing one. Precedence: whatever the pipeline attached (a generated soundtrack, or the driving speech an audio-conditioned family consumed) beats <see cref="VideoRequest.VideoAudioInput"/>, the caller-supplied pass-through track. <see cref="VideoRequest.VideoAudioReference"/> is deliberately NOT a fallback — it is conditioning, and a family that means it to be heard attaches it itself.</summary>
public static class VideoAudioResolver
{
    // Sub-frame shortfalls are latent-count rounding and pad silently; a quarter second means the wrong track.
    private const double ShortTrackWarnSeconds = 0.25d;

    /// <summary>Picks the track for <paramref name="result"/> and trims it to <paramref name="videoSeconds"/>; returns the result unchanged when there is nothing to attach.</summary>
    public static VideoGenerationResult Resolve(VideoGenerationResult result, VideoRequest request, double videoSeconds)
    {
        ArgumentNullException.ThrowIfNull(result);
        ArgumentNullException.ThrowIfNull(request);
        AudioBuffer? chosen = result.Audio is not null && !result.Audio.IsEmpty
            ? result.Audio
            : DecodePassThrough(request);
        if (chosen is null || chosen.IsEmpty)
        {
            return result.Audio is null ? result : result with { Audio = null };
        }
        if (videoSeconds <= 0d)
        {
            return result with { Audio = chosen };
        }
        // Muxers cut to the shorter stream (ffmpeg -shortest), so a track even one sample short drops trailing VIDEO
        // frames. Match the clip exactly: trim a long track, silence-pad a short one.
        AudioBuffer fitted = chosen.TrimTo(videoSeconds);
        double shortfall = videoSeconds - fitted.Seconds;
        fitted = fitted.PadTo(videoSeconds);
        if (shortfall > ShortTrackWarnSeconds)
        {
            Logs.Warning($"[VideoAudio] The soundtrack was {shortfall:0.00}s shorter than the {videoSeconds:0.00}s video "
                + "and has been silence-padded; check that the track belongs to this clip.");
        }
        return result with { Audio = fitted.IsEmpty ? null : fitted };
    }

    /// <summary>Seconds of video <paramref name="frameCount"/> frames occupy at <paramref name="fps"/>.</summary>
    public static double VideoSeconds(int frameCount, int? fps) => fps is > 0 ? frameCount / (double)fps.Value : 0d;

    private static AudioBuffer? DecodePassThrough(VideoRequest request)
    {
        if (request.VideoAudioInput is null)
        {
            return null;
        }
        try
        {
            return AudioClipCodec.DecodeNative(request.VideoAudioInput);
        }
        catch (Exception ex)
        {
            Logs.Error("[VideoAudio] Failed to decode VideoAudioInput for mux pass-through; continuing without a track.", ex);
            return null;
        }
    }
}
