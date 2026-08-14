using System.Diagnostics;
using HartsyInference.Core.Logging;

namespace HartsyInference.Video.Encoding;

/// <summary>Muxes an already-written PNG frame sequence, plus an optional soundtrack, into a playable H.264 MP4 by
/// invoking <c>ffmpeg</c>. Unlike <see cref="FfmpegProcessEncoder"/> this reads the frames back off disk rather than
/// streaming raw RGB, so it can run after the frames are written and can carry an audio track.</summary>
public static class FfmpegMuxer
{
    /// <summary>Writes <paramref name="outputPath"/> from <c>frame_%04d.png</c> in <paramref name="frameDir"/>.
    /// Returns false (having logged) when ffmpeg is absent or fails — the frames on disk remain the source of
    /// truth, so a missing container is a degraded result, never a failed generation.</summary>
    public static bool TryMux(string frameDir, int fps, string? audioWavPath, string outputPath, string ffmpegPath = "ffmpeg")
    {
        string args = $"-y -framerate {fps} -i \"{Path.Combine(frameDir, "frame_%04d.png")}\"";
        bool hasAudio = !string.IsNullOrEmpty(audioWavPath) && File.Exists(audioWavPath);
        if (hasAudio)
            args += $" -i \"{audioWavPath}\"";
        args += " -c:v libx264 -pix_fmt yuv420p -crf 18";
        if (hasAudio)
            args += " -c:a aac -b:a 192k -shortest";
        args += $" \"{outputPath}\"";

        ProcessStartInfo psi = new()
        {
            FileName = ffmpegPath,
            Arguments = args,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        try
        {
            using Process proc = Process.Start(psi) ?? throw new InvalidOperationException("Process.Start returned null.");
            string stderr = proc.StandardError.ReadToEnd();
            proc.WaitForExit();
            if (proc.ExitCode != 0)
            {
                Logs.Warning($"ffmpeg exited {proc.ExitCode} muxing {outputPath}; PNG frames were written. {stderr}");
                return false;
            }
            Logs.Info($"Muxed {outputPath}{(hasAudio ? " (with audio)" : "")}");
            return true;
        }
        catch (Exception ex)
        {
            Logs.Warning($"MP4 mux skipped (ffmpeg unavailable?) — PNG frames were written. {ex.Message}");
            return false;
        }
    }
}
