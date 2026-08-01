using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>Decides which waveform ships with a generated video, so every family gets the same behaviour instead of
/// each pipeline inventing one. Precedence: whatever the pipeline attached (a generated soundtrack, or the driving
/// speech an audio-conditioned family consumed) beats <see cref="VideoRequest.VideoAudioInput"/>, the caller-supplied
/// pass-through track. <see cref="VideoRequest.VideoAudioReference"/> is deliberately NOT a fallback — it is
/// conditioning, and a family that means it to be heard attaches it itself.</summary>
public static class VideoAudioResolver
{
    // One frame at 24fps; rounding in a model's audio-latent count is normal, a real shortfall is not.
    private const double ShortTrackToleranceSeconds = 0.05d;

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
        AudioBuffer trimmed = videoSeconds > 0d ? chosen.TrimTo(videoSeconds) : chosen;
        // Muxers cut to the shorter stream, so a short track silently truncates the VIDEO — say so rather than lose frames quietly.
        if (videoSeconds > 0d && trimmed.Seconds < videoSeconds - ShortTrackToleranceSeconds)
        {
            Logs.Warning($"[VideoAudio] The soundtrack is {trimmed.Seconds:0.00}s but the video is {videoSeconds:0.00}s; "
                + "a muxer that cuts to the shorter stream will drop the trailing frames.");
        }
        return result with { Audio = trimmed.IsEmpty ? null : trimmed };
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
