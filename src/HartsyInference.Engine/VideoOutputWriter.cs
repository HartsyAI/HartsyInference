using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Requests;
using HartsyInference.Video.Encoding;

namespace HartsyInference.Engine;

/// <summary>Writes a finished video generation to disk: numbered PNG frames, the soundtrack beside them, and a
/// playable <c>video.mp4</c> muxed from both. Shared by the CLI and the HTTP API so a generation looks the same
/// whichever one produced it.</summary>
public static class VideoOutputWriter
{
    /// <summary>Frame directory written under <paramref name="baseDir"/>, plus the muxed container's path when
    /// ffmpeg produced one.</summary>
    public readonly record struct Written(string Directory, string? Mp4Path, string? AudioPath);

    /// <summary>Writes frames, then <c>audio.wav</c> when <paramref name="audio"/> carries a soundtrack, then muxes
    /// <c>video.mp4</c>. A failed mux still returns the directory — the frames are the durable output.</summary>
    public static Written Write(byte[][] frames, int width, int height, string baseDir, string slug,
        AudioBuffer? audio, int fps)
    {
        string dir = FrameWriter.WriteFrames(frames, width, height, baseDir, slug);
        string? audioPath = null;
        if (audio is not null && !audio.IsEmpty)
        {
            audioPath = Path.Combine(dir, "audio.wav");
            File.WriteAllBytes(audioPath, AudioClipCodec.EncodeWav(audio));
        }
        string mp4Path = Path.Combine(dir, "video.mp4");
        bool muxed = FfmpegMuxer.TryMux(dir, fps <= 0 ? 24 : fps, audioPath, mp4Path);
        return new Written(dir, muxed ? mp4Path : null, audioPath);
    }
}
