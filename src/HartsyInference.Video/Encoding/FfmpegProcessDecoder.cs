using System.Diagnostics;
using System.Globalization;

namespace HartsyInference.Video.Encoding;

/// <summary>Decodes a video container to raw RGB24 frames by shelling out to <c>ffprobe</c>/<c>ffmpeg</c>
/// (child processes — the pure-C# rule bans native LIBRARIES, and <see cref="FfmpegProcessEncoder"/> set
/// the subprocess precedent). Throws a clear install hint when the binaries are missing. Uses PATH binaries
/// by default, or the folder named by the <c>paths.ffmpegDir</c> setting (see <see cref="FfmpegLocator"/>).</summary>
public sealed class FfmpegProcessDecoder(string? ffmpegPath = null, string? ffprobePath = null)
{
    private readonly string _ffmpegPath = ffmpegPath ?? FfmpegLocator.Resolve("ffmpeg");
    private readonly string _ffprobePath = ffprobePath ?? FfmpegLocator.Resolve("ffprobe");

    /// <summary>Decoded clip: frames are row-major RGB24, all the same size.</summary>
    public sealed record Result(List<byte[]> Frames, int Width, int Height, double Fps);

    /// <summary>Decodes container bytes (written to a temp file — many containers need seekable input).</summary>
    public Task<Result> DecodeAsync(byte[] container, string? formatHint, CancellationToken cancel = default) =>
        DecodeAsync(container, formatHint, maxFrames: null, scaleWidth: null, scaleHeight: null, cancel);

    /// <summary>Container-bytes variant of <see cref="DecodeFileAsync(string, int?, int?, int?, CancellationToken)"/>:
    /// the frame cap and ffmpeg-side scale run before frames cross the pipe, so a long HD clip never materializes.</summary>
    public async Task<Result> DecodeAsync(byte[] container, string? formatHint, int? maxFrames, int? scaleWidth,
        int? scaleHeight, CancellationToken cancel = default, bool letterbox = false, double? maxSeconds = null)
    {
        string temp = Path.Combine(Path.GetTempPath(), $"hartsy-dec-{Guid.NewGuid():N}.{SafeExtension(formatHint)}");
        await File.WriteAllBytesAsync(temp, container, cancel).ConfigureAwait(false);
        try
        {
            return await DecodeFileAsync(temp, maxFrames, scaleWidth, scaleHeight, cancel, letterbox, maxSeconds).ConfigureAwait(false);
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
    /// scale dimension keeps the probed source size. <paramref name="maxSeconds"/> stops decoding after that much input.</summary>
    public async Task<Result> DecodeFileAsync(string path, int? maxFrames, int? scaleWidth, int? scaleHeight,
        CancellationToken cancel = default, bool letterbox = false, double? maxSeconds = null)
    {
        if (maxFrames is <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxFrames), $"maxFrames must be positive; got {maxFrames}.");
        if (maxSeconds is <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxSeconds), $"maxSeconds must be positive; got {maxSeconds}.");
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
        string durationArg = maxSeconds is null ? "" : string.Create(CultureInfo.InvariantCulture, $"-t {maxSeconds.Value} ");

        ProcessStartInfo psi = new()
        {
            FileName = _ffmpegPath,
            Arguments = $"-v error {durationArg}-i \"{path}\"{scaleArg}{capArg} -f rawvideo -pix_fmt rgb24 -",
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

    /// <summary>A video stream's size, frame rate and frame count.</summary>
    /// <param name="FrameCount">The container's frame count, or duration × fps when the container does not record it.</param>
    public sealed record StreamInfo(int Width, int Height, double Fps, int FrameCount);

    /// <summary>Reads the first video stream's size, frame rate and frame count without decoding it.</summary>
    public async Task<StreamInfo> ProbeStreamAsync(string path, CancellationToken cancel = default)
    {
        (int width, int height, double fps) = await ProbeAsync(path, cancel).ConfigureAwait(false);
        string counts = await RunProbeAsync(
            ["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames:format=duration", "-of", "default=nw=1", path],
            cancel).ConfigureAwait(false);
        int frames = 0;
        double duration = 0;
        foreach (string line in counts.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            if (line.StartsWith("nb_frames=", StringComparison.Ordinal))
                int.TryParse(line.AsSpan(10), NumberStyles.Integer, CultureInfo.InvariantCulture, out frames);
            else if (line.StartsWith("duration=", StringComparison.Ordinal))
                double.TryParse(line.AsSpan(9), NumberStyles.Float, CultureInfo.InvariantCulture, out duration);
        }
        if (frames <= 0) frames = (int)Math.Round(duration * fps);
        if (frames <= 0)
            throw new InvalidOperationException($"ffprobe reported neither a frame count nor a duration for '{path}'.");
        return new StreamInfo(width, height, fps, frames);
    }

    /// <summary>Streams the first video stream's frames at source size as RGB24, in decode order, without holding the clip.
    /// Each yielded buffer is reused for the next frame: copy what you keep before advancing. Only the first video stream is
    /// decoded (<c>-map 0:v:0</c>, matching the probe) and rotation metadata is ignored (<c>-noautorotate</c>), so every frame
    /// has the probed <paramref name="width"/>×<paramref name="height"/>.</summary>
    /// <exception cref="InvalidOperationException">ffmpeg exits non-zero, or its output ends inside a frame.</exception>
    public async IAsyncEnumerable<ReadOnlyMemory<byte>> StreamFramesAsync(string path, int width, int height,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancel = default)
    {
        ProcessStartInfo psi = NewStartInfo(_ffmpegPath,
            ["-v", "error", "-noautorotate", "-i", path, "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]);
        using Process proc = Start(psi, _ffmpegPath);
        Task<string> stderrTask = proc.StandardError.ReadToEndAsync(cancel);
        int frameBytes = checked(width * height * 3);
        byte[] buffer = new byte[frameBytes];
        Stream stdout = proc.StandardOutput.BaseStream;
        bool completed = false;
        try
        {
            while (true)
            {
                int filled = 0;
                while (filled < frameBytes)
                {
                    int read = await stdout.ReadAsync(buffer.AsMemory(filled, frameBytes - filled), cancel).ConfigureAwait(false);
                    if (read == 0) break;
                    filled += read;
                }
                if (filled == 0) break;
                if (filled < frameBytes)
                    throw new InvalidOperationException(
                        $"ffmpeg output for '{path}' ended inside a frame ({filled} of {frameBytes} bytes); the stream size changed mid-clip?");
                yield return buffer;
            }
            await proc.WaitForExitAsync(cancel).ConfigureAwait(false);
            string stderr = await stderrTask.ConfigureAwait(false);
            if (proc.ExitCode != 0)
                throw new InvalidOperationException($"ffmpeg decode exited {proc.ExitCode}: {stderr}");
            completed = true;
        }
        finally
        {
            // The consumer may stop early (it has every frame it planned); do not leave ffmpeg blocked on a full pipe.
            if (!completed) KillQuietly(proc);
            // Observe the stderr reader so an early exit does not leave an unobserved faulted task behind.
            _ = stderrTask.ContinueWith(static t => _ = t.Exception, TaskScheduler.Default);
        }
    }

    private static void KillQuietly(Process proc)
    {
        try
        {
            if (proc.HasExited) return;
            proc.Kill(entireProcessTree: true);
            proc.WaitForExit(1000);
        }
        catch (Exception ex) when (ex is InvalidOperationException or System.ComponentModel.Win32Exception or NotSupportedException)
        {
            // Already gone, or not ours to kill: nothing left to clean up.
        }
    }

    private static ProcessStartInfo NewStartInfo(string binary, IEnumerable<string> arguments)
    {
        ProcessStartInfo psi = new()
        {
            FileName = binary,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        // ArgumentList quotes each entry for the platform, so a path holding quotes or spaces stays one argument.
        foreach (string argument in arguments) psi.ArgumentList.Add(argument);
        return psi;
    }

    /// <summary>A container extension safe to put in a temp file name: ASCII letters and digits only, else <c>bin</c>.</summary>
    internal static string SafeExtension(string? formatHint)
    {
        if (string.IsNullOrEmpty(formatHint) || formatHint.Length > 16) return "bin";
        foreach (char c in formatHint)
            if (!char.IsAsciiLetterOrDigit(c)) return "bin";
        return formatHint;
    }

    private async Task<string> RunProbeAsync(IEnumerable<string> arguments, CancellationToken cancel)
    {
        ProcessStartInfo psi = NewStartInfo(_ffprobePath, arguments);
        using Process proc = Start(psi, _ffprobePath);
        string output = await proc.StandardOutput.ReadToEndAsync(cancel).ConfigureAwait(false);
        await proc.StandardError.ReadToEndAsync(cancel).ConfigureAwait(false);
        await proc.WaitForExitAsync(cancel).ConfigureAwait(false);
        return output;
    }

    private async Task<(int Width, int Height, double Fps)> ProbeAsync(string path, CancellationToken cancel)
    {
        ProcessStartInfo psi = NewStartInfo(_ffprobePath,
            ["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0", path]);
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
                $"Failed to launch '{binary}'. Video input requires ffmpeg. {FfmpegLocator.InstallHint} {ex.Message}", ex);
        }
    }
}
