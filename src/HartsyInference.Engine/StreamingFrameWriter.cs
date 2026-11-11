using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using HartsyInference.Core.Numerics;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine;

/// <summary>Writes a video generation's frames to disk as they arrive, so a long or unbounded stream holds none of its pixels
/// in memory: each frame is encoded into its own numbered PNG, and the writer keeps only a count, the frame size and the first
/// frame (the artifact's preview). <see cref="Complete"/> then writes the soundtrack and muxes <c>video.mp4</c> from the frames on
/// disk, as <see cref="VideoOutputWriter.Write"/> does for a finished batch.</summary>
/// <remarks>
/// PNG encoding is the slow part of a frame (about 110 ms for a 1280x704 frame), so <see cref="Add"/> does not encode: it queues the
/// frame for a small pool of background writers, and returns. The queue holds at most <see cref="QueueCapacity"/> frames; past that,
/// <see cref="Add"/> waits for a writer, so a slow disk or encoder holds the stream back rather than letting memory grow. The writers
/// encode in parallel, and every file is named by its frame's position, so the files are the same bytes a sequential writer produces.
/// <see cref="WaitForWrites"/> blocks until every frame added so far is on disk. <see cref="Complete"/> waits for them, and so does
/// <see cref="Dispose"/>, so a stream that stops early still leaves every frame it produced on disk.
/// </remarks>
public sealed class StreamingFrameWriter : IDisposable
{
    /// <summary>Frames that may wait for a writer before <see cref="Add"/> blocks: about 90 MB at 1280x704.</summary>
    public const int QueueCapacity = 32;

    private readonly string _baseDir;
    private readonly string _slug;
    private readonly ManualResetEventSlim? _writeGate;
    private readonly object _gate = new();
    private readonly ManualResetEventSlim _idle = new(initialState: true);
    private BlockingCollection<PendingFrame>? _queue;
    private Task[] _writers = [];
    private int _outstanding;
    private Exception? _failure;
    private bool _closed;
    private string? _outputDirectory;
    private byte[]? _firstFrame;

    /// <summary>Creates a writer that will put its frames in a new subdirectory of <paramref name="baseDir"/> named after
    /// <paramref name="slug"/>.</summary>
    public StreamingFrameWriter(string baseDir, string slug)
        : this(baseDir, slug, writeGate: null)
    {
    }

    /// <summary>Test seam: when <paramref name="writeGate"/> is set, every writer waits on it before writing, so a test can hold the
    /// writers back and watch the queue fill.</summary>
    internal StreamingFrameWriter(string baseDir, string slug, ManualResetEventSlim? writeGate)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(baseDir);
        ArgumentNullException.ThrowIfNull(slug);
        _baseDir = baseDir;
        _slug = slug;
        _writeGate = writeGate;
    }

    /// <summary>The directory the frames are written to, or null until the first frame arrives.</summary>
    public string? OutputDirectory => _outputDirectory;

    /// <summary>Frames accepted so far. A frame is counted when it is queued, so it may not be on disk yet; see <see cref="WaitForWrites"/>.</summary>
    public int Count { get; private set; }

    /// <summary>Frame width in pixels, or 0 before the first frame.</summary>
    public int Width { get; private set; }

    /// <summary>Frame height in pixels, or 0 before the first frame.</summary>
    public int Height { get; private set; }

    /// <summary>The first frame's RGB24 pixels, kept as the artifact's preview, or null before the first frame. It is the only
    /// frame the writer holds once that frame is on disk.</summary>
    public byte[]? FirstFrame => _firstFrame;

    /// <summary>Queues <paramref name="rgb"/> (<c>width*height*3</c> bytes) as the next frame. It is encoded and written by a background
    /// writer; the caller must not change the array afterwards. Blocks while <see cref="QueueCapacity"/> frames are waiting.</summary>
    /// <exception cref="ArgumentException">The pixels do not fill <paramref name="width"/> by <paramref name="height"/>, or the
    /// size differs from the first frame's.</exception>
    /// <exception cref="InvalidOperationException">The writer is complete, or an earlier frame could not be written.</exception>
    public void Add(byte[] rgb, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(rgb);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        if (rgb.LongLength != (long)width * height * 3)
            throw new ArgumentException($"Frame {Count + 1} holds {rgb.LongLength} bytes; {width}x{height} RGB24 needs {(long)width * height * 3}.",
                nameof(rgb));
        if (Count > 0 && (width != Width || height != Height))
            throw new ArgumentException($"Frame {Count + 1} is {width}x{height}; the stream began at {Width}x{Height}.", nameof(width));
        lock (_gate)
        {
            if (_closed)
                throw new InvalidOperationException("The frame writer is complete; no frame can follow.");
            if (_failure is not null)
                throw new InvalidOperationException($"A frame could not be written: {_failure.Message}", _failure);
        }

        if (_queue is null)
            StartWriters(FrameWriter.CreateDirectory(_baseDir, _slug));
        if (Count == 0)
        {
            _firstFrame = rgb;
            Width = width;
            Height = height;
        }
        PendingFrame pending = new() { Index = Count, Rgb = rgb, Width = width, Height = height };
        Count++;
        lock (_gate)
        {
            _outstanding++;
            _idle.Reset();
        }
        _queue!.Add(pending);
    }

    /// <summary>Blocks until every frame added so far is on disk. Throws if a frame could not be written.</summary>
    public void WaitForWrites()
    {
        _idle.Wait();
        ThrowIfFailed();
    }

    /// <summary>Finishes the video: waits for the frames, then writes <c>audio.wav</c> when <paramref name="audio"/> carries a soundtrack,
    /// and <c>video.mp4</c> muxed from the frames on disk (see <see cref="VideoOutputWriter.Finish"/>).</summary>
    /// <exception cref="InvalidOperationException">No frame was added, or a frame could not be written.</exception>
    public VideoOutputWriter.Written Complete(AudioBuffer? audio, int fps)
    {
        if (_outputDirectory is null)
            throw new InvalidOperationException("No frames were written.");
        CloseWriters();
        ThrowIfFailed();
        return VideoOutputWriter.Finish(_outputDirectory, audio, fps);
    }

    /// <summary>Stops accepting frames and waits for the queued ones to reach disk. Safe to call after <see cref="Complete"/>.</summary>
    public void Dispose() => CloseWriters();

    private void StartWriters(string directory)
    {
        _outputDirectory = directory;
        BlockingCollection<PendingFrame> queue = new(QueueCapacity);
        // A quarter of the CPU budget, so the writers leave the rest to the model and to audio; at most four.
        int writers = Math.Clamp(CpuParallel.MaxThreads / 4, 1, 4);
        Task[] tasks = new Task[writers];
        for (int i = 0; i < writers; i++)
        {
            tasks[i] = Task.Factory.StartNew(() => WriteFrames(queue, directory), CancellationToken.None,
                TaskCreationOptions.LongRunning, TaskScheduler.Default);
        }
        _writers = tasks;
        _queue = queue;
    }

    private void WriteFrames(BlockingCollection<PendingFrame> queue, string directory)
    {
        foreach (PendingFrame pending in queue.GetConsumingEnumerable())
        {
            WriteOne(pending, directory);
            lock (_gate)
            {
                if (--_outstanding == 0)
                {
                    _idle.Set();
                }
            }
        }
    }

    // Kept out of the loop so the pixels are referenced only by this call's frame: a stale copy in the loop's own frame would keep
    // the last frame alive until the next one arrives.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private void WriteOne(PendingFrame pending, string directory)
    {
        try
        {
            bool failed;
            lock (_gate)
            {
                failed = _failure is not null;
            }
            if (!failed)
            {
                _writeGate?.Wait();
                FrameWriter.WriteFrame(directory, pending.Index, pending.Rgb!, pending.Width, pending.Height);
            }
        }
        catch (Exception ex)
        {
            lock (_gate)
            {
                _failure ??= ex;
            }
        }
        finally
        {
            // The frame is no longer the writer's to hold: clearing it frees the pixels as soon as they are on disk.
            pending.Rgb = null;
        }
    }

    private void CloseWriters()
    {
        lock (_gate)
        {
            if (_closed)
            {
                return;
            }
            _closed = true;
        }
        _queue?.CompleteAdding();
        Task.WaitAll(_writers);
    }

    private void ThrowIfFailed()
    {
        Exception? failure;
        lock (_gate)
        {
            failure = _failure;
        }
        if (failure is not null)
            throw new InvalidOperationException($"A frame could not be written: {failure.Message}", failure);
    }

    /// <summary>A frame waiting for a writer. The writer clears <see cref="Rgb"/> once the frame is on disk, so the queue entry never keeps
    /// the pixels alive.</summary>
    private sealed class PendingFrame
    {
        public int Index { get; init; }
        public byte[]? Rgb { get; set; }
        public int Width { get; init; }
        public int Height { get; init; }
    }
}
