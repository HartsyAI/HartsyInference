using System.Diagnostics;
using System.Globalization;

namespace HartsyInference.Video.Encoding;

/// <summary>Decodes a video container to raw RGB24 frames by shelling out to <c>ffprobe</c>/<c>ffmpeg</c>
/// (child processes — the pure-C# rule bans native LIBRARIES, and <see cref="FfmpegProcessEncoder"/> set
/// the subprocess precedent). Throws a clear install hint when the binaries are missing.</summary>
public sealed class FfmpegProcessDecoder
{
    private readonly string _ffmpegPath;
    private readonly string _ffprobePath;

    /// <summary>Uses PATH binaries by default.</summary>
    public FfmpegProcessDecoder(string ffmpegPath = "ffmpeg", string ffprobePath = "ffprobe")
    {
        _ffmpegPath = ffmpegPath;
        _ffprobePath = ffprobePath;
    }

    /// <summary>Decoded clip: frames are row-major RGB24, all the same size.</summary>
    public sealed record Result(List<byte[]> Frames, int Width, int Height, double Fps);

    /// <summary>Decodes container bytes (written to a temp file — many containers need seekable input).</summary>
    public Task<Result> DecodeAsync(byte[] container, string? formatHint, CancellationToken cancel = default) =>
        DecodeAsync(container, formatHint, maxFrames: null, scaleWidth: null, scaleHeight: null, cancel);

    /// <summary>Container-bytes variant of <see cref="DecodeFileAsync(string, int?, int?, int?, CancellationToken)"/>:
    /// the frame cap and ffmpeg-side scale run before frames cross the pipe, so a long HD clip never materializes.</summary>
    public async Task<Result> DecodeAsync(byte[] container, string? formatHint, int? maxFrames, int? scaleWidth,
        int? scaleHeight, CancellationToken cancel = default, bool letterbox = false)
    {
        string temp = Path.Combine(Path.GetTempPath(), $"hartsy-dec-{Guid.NewGuid():N}.{formatHint ?? "bin"}");
        await File.WriteAllBytesAsync(temp, container, cancel).ConfigureAwait(false);
        try
        {
            return await DecodeFileAsync(temp, maxFrames, scaleWidth, scaleHeight, cancel, letterbox).ConfigureAwait(false);
        }
        finally
        {
            try { File.Delete(temp); } catch (IOException) { }
        }
    }

    /// <summary>Decodes a video file on disk.</summary>
    public Task<Result> DecodeFileAsync(string path, CancellationToken cancel = default) =>
        DecodeFileAsync(path, maxFrames: null, scaleWidth: null, scaleHeight: null, cancel);

    /// <summary>Decodes a video file on disk, optionally capped to <paramref name="maxFrames"/> (<c>-frames:v</c>) and
    /// resampled ffmpeg-side to <paramref name="scaleWidth"/>×<paramref name="scaleHeight"/> (<c>-vf scale</c>); a null
    /// scale dimension keeps the probed source size.</summary>
    public async Task<Result> DecodeFileAsync(string path, int? maxFrames, int? scaleWidth, int? scaleHeight,
        CancellationToken cancel = default, bool letterbox = false)
    {
        if (maxFrames is <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxFrames), $"maxFrames must be positive; got {maxFrames}.");
        if (scaleWidth is <= 0 || scaleHeight is <= 0)
            throw new ArgumentOutOfRangeException(nameof(scaleWidth), $"Scale dimensions must be positive; got {scaleWidth}x{scaleHeight}.");
        (int width, int height, double fps) = await ProbeAsync(path, cancel).ConfigureAwait(false);
        int outWidth = scaleWidth ?? width;
        int outHeight = scaleHeight ?? height;
        // A bare scale=W:H shears the source when the aspects differ. Letterboxing (fit inside, centre on black) is
        // what Wan2.2's padding_resize does, and the pose skeleton it produces has to land on the body.
        string filter = letterbox
            ? $"scale={outWidth}:{outHeight}:force_original_aspect_ratio=decrease,"
                + $"pad={outWidth}:{outHeight}:(ow-iw)/2:(oh-ih)/2:black"
            : $"scale={outWidth}:{outHeight}";
        string scaleArg = scaleWidth is null && scaleHeight is null ? "" : $" -vf \"{filter}\"";
        string capArg = maxFrames is null ? "" : $" -frames:v {maxFrames.Value}";

        ProcessStartInfo psi = new()
        {
            FileName = _ffmpegPath,
            Arguments = $"-v error -i \"{path}\"{scaleArg}{capArg} -f rawvideo -pix_fmt rgb24 -",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        using Process proc = Start(psi, _ffmpegPath);

        int frameBytes = outWidth * outHeight * 3;
        List<byte[]> frames = new();
        byte[] buffer = new byte[frameBytes];
        int filled = 0;
        Stream stdout = proc.StandardOutput.BaseStream;
        Task<string> stderrTask = proc.StandardError.ReadToEndAsync(cancel);
        while (true)
        {
            int read = await stdout.ReadAsync(buffer.AsMemory(filled, frameBytes - filled), cancel).ConfigureAwait(false);
            if (read == 0)
                break;
            filled += read;
            if (filled == frameBytes)
            {
                frames.Add(buffer);
                buffer = new byte[frameBytes];
                filled = 0;
            }
        }
        await proc.WaitForExitAsync(cancel).ConfigureAwait(false);
        if (proc.ExitCode != 0)
            throw new InvalidOperationException($"ffmpeg decode exited {proc.ExitCode}: {await stderrTask.ConfigureAwait(false)}");
        if (frames.Count == 0)
            throw new InvalidOperationException($"ffmpeg produced no frames for '{path}' ({outWidth}x{outHeight}).");
        return new Result(frames, outWidth, outHeight, fps);
    }

    private async Task<(int Width, int Height, double Fps)> ProbeAsync(string path, CancellationToken cancel)
    {
        ProcessStartInfo psi = new()
        {
            FileName = _ffprobePath,
            Arguments = $"-v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0 \"{path}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        using Process proc = Start(psi, _ffprobePath);
        string output = (await proc.StandardOutput.ReadToEndAsync(cancel).ConfigureAwait(false)).Trim();
        string stderr = await proc.StandardError.ReadToEndAsync(cancel).ConfigureAwait(false);
        await proc.WaitForExitAsync(cancel).ConfigureAwait(false);
        if (proc.ExitCode != 0 || output.Length == 0)
            throw new InvalidOperationException($"ffprobe failed for '{path}': {stderr}");

        string[] parts = output.Split(',');
        int width = int.Parse(parts[0], CultureInfo.InvariantCulture);
        int height = int.Parse(parts[1], CultureInfo.InvariantCulture);
        double fps = 25.0;
        if (parts.Length > 2 && parts[2].Contains('/'))
        {
            string[] ratio = parts[2].Split('/');
            double num = double.Parse(ratio[0], CultureInfo.InvariantCulture);
            double den = double.Parse(ratio[1], CultureInfo.InvariantCulture);
            if (den > 0 && num > 0)
                fps = num / den;
        }
        return (width, height, fps);
    }

    private static Process Start(ProcessStartInfo psi, string binary)
    {
        try
        {
            return Process.Start(psi) ?? throw new InvalidOperationException($"'{binary}' failed to start.");
        }
        catch (Exception ex) when (ex is not InvalidOperationException)
        {
            throw new InvalidOperationException(
                $"Failed to launch '{binary}'. Video restoration input requires ffmpeg — install it (e.g. apt install ffmpeg). {ex.Message}", ex);
        }
    }
}
