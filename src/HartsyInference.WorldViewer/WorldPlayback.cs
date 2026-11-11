using System.Diagnostics;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Worlds;

namespace HartsyInference.WorldViewer;

/// <summary>A bounded buffer between a frame stream and a presenter. The stream writes frames as the session delivers them and waits
/// when the buffer is full, so a slow viewer holds generation back rather than letting control lag grow without limit. The presenter
/// (on its UI thread) asks <see cref="Step"/> on every clock tick and shows the frame it returns.</summary>
/// <remarks>Writes come from one stream thread and <see cref="Step"/> and <see cref="Status"/> from one presenter thread. Frames are
/// not copied: each frame's pixels are a fresh allocation (<c>SanaWmStream.ToRgb24</c>), so the buffer owns them outright.</remarks>
public sealed class WorldPlayback
{
    private readonly object _gate = new();
    private readonly Queue<VideoFrame> _frames = new();
    private readonly SemaphoreSlim _space;
    private readonly WorldPacer _pacer;
    private readonly int _fps;
    private readonly int _capacity;
    private bool _complete;
    private string? _error;
    private long _received;
    private int _shown;
    private long _firstWriteTimestamp = -1;
    private long _completeTimestamp = -1;

    /// <summary>How many frames a window may run ahead of the frames it has shown. A stream that takes one action per step gets two
    /// frames, so a key press reaches the picture after about one step; any other stream gets two seconds of video.</summary>
    public static int DefaultBufferFrames(int fps, WorldControlCadence? cadence)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(fps, 1);
        return cadence == WorldControlCadence.PerStep ? PerStepBufferFrames : 2 * fps;
    }

    /// <summary>Frames a per-step stream may run ahead of the window: the buffer is the control lag, so it is kept to two frames.</summary>
    public const int PerStepBufferFrames = 2;

    /// <summary>Creates a buffer that holds <paramref name="bufferFrames"/> frames and plays at <paramref name="fps"/>. Playback starts
    /// after one second of frames, or the whole buffer if it is smaller.</summary>
    public WorldPlayback(int fps, int bufferFrames)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(fps, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(bufferFrames, 1);
        _fps = fps;
        _capacity = bufferFrames;
        int start = Math.Min(fps, bufferFrames);
        int resume = Math.Min(Math.Max(1, fps / 4), start);
        _pacer = new WorldPacer(fps, start, resume);
        _space = new SemaphoreSlim(bufferFrames, bufferFrames);
    }

    /// <summary>Waits for room, then buffers <paramref name="frame"/>.</summary>
    public async ValueTask WriteAsync(VideoFrame frame, CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(frame);
        lock (_gate)
        {
            if (_complete)
            {
                throw new InvalidOperationException("The playback buffer is complete; no frame can follow.");
            }
        }
        await _space.WaitAsync(cancel).ConfigureAwait(false);
        lock (_gate)
        {
            if (_firstWriteTimestamp < 0)
            {
                _firstWriteTimestamp = Stopwatch.GetTimestamp();
            }
            _frames.Enqueue(frame);
            _received++;
        }
    }

    /// <summary>Records that the stream delivered its last frame, or ended with <paramref name="error"/> when it failed.</summary>
    public void Complete(string? error = null)
    {
        lock (_gate)
        {
            if (!_complete)
            {
                // The delivered rate is measured up to here, so a window that stays open after the run reads the run's rate.
                _completeTimestamp = Stopwatch.GetTimestamp();
            }
            _complete = true;
            _error ??= error;
        }
    }

    /// <summary>Decides at <paramref name="now"/> whether to show a frame; on <see cref="WorldPaceStep.Show"/> the frame is returned.</summary>
    public WorldPaceStep Step(TimeSpan now, out VideoFrame? frame)
    {
        int buffered;
        bool ended;
        lock (_gate)
        {
            buffered = _frames.Count;
            ended = _complete;
        }
        WorldPaceStep step = _pacer.Step(now, buffered, ended);
        frame = null;
        if (step != WorldPaceStep.Show)
        {
            return step;
        }
        lock (_gate)
        {
            frame = _frames.Dequeue();
            _shown++;
        }
        _space.Release();
        return step;
    }

    /// <summary>A snapshot of playback as of the last <see cref="Step"/>.</summary>
    public WorldPlaybackStatus Status()
    {
        int buffered;
        bool complete;
        string? error;
        long received;
        int shown;
        long firstWrite;
        long completed;
        lock (_gate)
        {
            buffered = _frames.Count;
            complete = _complete;
            error = _error;
            received = _received;
            shown = _shown;
            firstWrite = _firstWriteTimestamp;
            completed = _completeTimestamp;
        }
        WorldPlaybackPhase phase;
        if (error is not null)
        {
            phase = WorldPlaybackPhase.Failed;
        }
        else if (!_pacer.Started)
        {
            phase = complete ? WorldPlaybackPhase.Finished : WorldPlaybackPhase.Generating;
        }
        else if (complete && buffered == 0)
        {
            phase = WorldPlaybackPhase.Finished;
        }
        else
        {
            phase = _pacer.Playing ? WorldPlaybackPhase.Playing : WorldPlaybackPhase.Rebuffering;
        }
        double elapsed = firstWrite < 0 ? 0
            : (completed >= 0 ? Stopwatch.GetElapsedTime(firstWrite, completed) : Stopwatch.GetElapsedTime(firstWrite)).TotalSeconds;
        double delivered = elapsed > 0 ? received / elapsed : 0;
        return new WorldPlaybackStatus(phase, _fps, buffered, _capacity, shown, received, delivered, error);
    }
}
