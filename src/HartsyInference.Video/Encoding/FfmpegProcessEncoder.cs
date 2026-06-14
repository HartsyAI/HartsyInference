using System.Diagnostics;
using HartsyInference.Core.Logging;

namespace HartsyInference.Video.Encoding;

/// <summary>Encodes a frame stream to an H.264 MP4 by piping raw RGB24 to an <c>ffmpeg</c> child process. Requires ffmpeg on PATH (or pass an explicit path). Throws a clear error if ffmpeg can't be launched — callers can fall back to <see cref="BmpSequenceEncoder"/>.</summary>
public sealed class FfmpegProcessEncoder : IVideoEncoder
{
    private readonly string _ffmpegPath;

    public FfmpegProcessEncoder(string ffmpegPath = "ffmpeg") => _ffmpegPath = ffmpegPath;

    /// <inheritdoc/>
    public async Task EncodeAsync(IAsyncEnumerable<VideoFrame> frames, string outputPath, int fps, CancellationToken cancellationToken = default)
    {
        await using IAsyncEnumerator<VideoFrame> e = frames.GetAsyncEnumerator(cancellationToken);
        if (!await e.MoveNextAsync())
            throw new InvalidOperationException("No frames to encode.");
        VideoFrame first = e.Current;

        string args = $"-y -f rawvideo -pixel_format rgb24 -video_size {first.Width}x{first.Height} " +
                      $"-framerate {fps} -i - -c:v libx264 -pix_fmt yuv420p \"{outputPath}\"";
        ProcessStartInfo psi = new()
        {
            FileName = _ffmpegPath,
            Arguments = args,
            RedirectStandardInput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        Process proc;
        try
        {
            proc = Process.Start(psi) ?? throw new InvalidOperationException("Process.Start returned null.");
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Failed to launch ffmpeg ('{_ffmpegPath}'). Install ffmpeg or use BmpSequenceEncoder. {ex.Message}", ex);
        }

        try
        {
            Stream stdin = proc.StandardInput.BaseStream;
            await stdin.WriteAsync(first.Rgb, cancellationToken);
            while (await e.MoveNextAsync())
                await stdin.WriteAsync(e.Current.Rgb, cancellationToken);
            stdin.Close();

            string stderr = await proc.StandardError.ReadToEndAsync(cancellationToken);
            await proc.WaitForExitAsync(cancellationToken);
            if (proc.ExitCode != 0)
                throw new InvalidOperationException($"ffmpeg exited {proc.ExitCode}: {stderr}");
            Logs.Info($"Encoded video → {outputPath}");
        }
        finally
        {
            if (!proc.HasExited) { try { proc.Kill(); } catch (Exception ex) { Logs.Error("Failed to kill ffmpeg", ex); } }
            proc.Dispose();
        }
    }
}
