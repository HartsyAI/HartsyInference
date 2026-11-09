using HartsyInference.Core.Imaging;

namespace HartsyInference.Video.Encoding;

/// <summary>A video file sampled for a vision-language model: <see cref="VideoFrameSampling.Plan"/> picks frame indices from the
/// stream's probed frame count and rate, and only those frames are kept while ffmpeg streams the clip, so a long video never
/// sits in memory. Frames come at source size; the model's own processor resizes them.</summary>
/// <remarks>With <c>paced</c> set, each frame is released no earlier than its timestamp after the first read, replaying the file
/// as a live camera would deliver it (for realtime sessions and demos).</remarks>
public sealed class FfmpegFrameSource : IVideoFrameSource, IDisposable
{
    private readonly string _path;
    private readonly bool _ownsFile;
    private readonly VideoFrameSamplingOptions _options;
    private readonly FfmpegProcessDecoder _decoder;
    private readonly bool _paced;
    private int _disposed;

    /// <summary>Samples the video file at <paramref name="path"/>.</summary>
    public FfmpegFrameSource(string path, VideoFrameSamplingOptions options, bool paced = false, FfmpegProcessDecoder? decoder = null)
        : this(path, ownsFile: false, options, paced, decoder)
    {
    }

    private FfmpegFrameSource(string path, bool ownsFile, VideoFrameSamplingOptions options, bool paced, FfmpegProcessDecoder? decoder)
    {
        ArgumentNullException.ThrowIfNull(options);
        _path = path;
        _ownsFile = ownsFile;
        _options = options;
        _paced = paced;
        _decoder = decoder ?? new FfmpegProcessDecoder();
    }

    /// <summary>Samples an in-memory container (written to a temporary file, deleted on <see cref="Dispose"/>; many containers
    /// need a seekable input).</summary>
    public static FfmpegFrameSource FromBytes(byte[] container, string? formatHint, VideoFrameSamplingOptions options, bool paced = false,
        FfmpegProcessDecoder? decoder = null)
    {
        string temp = Path.Combine(Path.GetTempPath(), $"hartsy-video-{Guid.NewGuid():N}.{formatHint ?? "bin"}");
        File.WriteAllBytes(temp, container);
        return new FfmpegFrameSource(temp, ownsFile: true, options, paced, decoder);
    }

    /// <summary>The frames the plan picks, in order, with their timestamps.</summary>
    public async IAsyncEnumerable<TimedRgbFrame> ReadAsync([System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancel = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        FfmpegProcessDecoder.StreamInfo info = await _decoder.ProbeStreamAsync(_path, cancel).ConfigureAwait(false);
        SampledFrame[] plan = VideoFrameSampling.Plan(info.FrameCount, info.Fps, _options);
        if (plan.Length == 0) yield break;

        int next = 0, index = 0;
        System.Diagnostics.Stopwatch clock = System.Diagnostics.Stopwatch.StartNew();
        await foreach (ReadOnlyMemory<byte> frame in _decoder.StreamFramesAsync(_path, info.Width, info.Height, cancel).ConfigureAwait(false))
        {
            if (index++ != plan[next].Index) continue;
            SampledFrame pick = plan[next++];
            if (_paced)
            {
                TimeSpan due = TimeSpan.FromSeconds(pick.TimestampSeconds) - clock.Elapsed;
                if (due > TimeSpan.Zero) await Task.Delay(due, cancel).ConfigureAwait(false);
            }
            yield return new TimedRgbFrame(new RawRgbFrame(frame.ToArray(), info.Width, info.Height), pick.TimestampSeconds);
            if (next == plan.Length) yield break;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0 || !_ownsFile) return;
        try { File.Delete(_path); } catch (IOException) { /* a leftover temp file is harmless */ }
    }
}
