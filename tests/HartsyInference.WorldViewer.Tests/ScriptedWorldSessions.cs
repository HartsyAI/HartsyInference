using System.Runtime.CompilerServices;
using HartsyInference.Engine.Requests;

namespace HartsyInference.WorldViewer.Tests;

/// <summary>A session that replays a script: actions are recorded, and each stream is whatever the test supplies.</summary>
internal sealed class ScriptedWorldSession : IWorldSession
{
    private readonly Func<CancellationToken, IAsyncEnumerable<VideoFrame>> _stream;
    private readonly object _gate = new();
    private readonly List<string> _actions = [];

    public ScriptedWorldSession(Func<CancellationToken, IAsyncEnumerable<VideoFrame>> stream)
    {
        _stream = stream;
    }

    public int StreamCalls { get; private set; }

    public bool Disposed { get; private set; }

    public IReadOnlyList<string> Actions
    {
        get
        {
            lock (_gate)
            {
                return _actions.ToArray();
            }
        }
    }

    public void SendAction(string action)
    {
        lock (_gate)
        {
            _actions.Add(action);
        }
    }

    public IAsyncEnumerable<VideoFrame> StreamAsync(CancellationToken cancel)
    {
        lock (_gate)
        {
            StreamCalls++;
        }
        return _stream(cancel);
    }

    public void Dispose() => Disposed = true;
}

/// <summary>A per-step session in the manner of DIAMOND: each queued action yields exactly one frame, and the stream ends when the
/// queue drains.</summary>
internal sealed class PerStepWorldSession : IWorldSession
{
    private readonly Queue<string> _pending = new();
    private readonly List<string> _taken = [];
    private int _index;

    public IReadOnlyList<string> Taken => _taken;

    public void SendAction(string action)
    {
        _pending.Enqueue(action);
    }

    public async IAsyncEnumerable<VideoFrame> StreamAsync([EnumeratorCancellation] CancellationToken cancel)
    {
        while (_pending.TryDequeue(out string? action))
        {
            cancel.ThrowIfCancellationRequested();
            _taken.Add(action);
            await Task.Yield();
            yield return TestFrames.Frame(_index++);
        }
    }

    public void Dispose()
    {
    }
}

/// <summary>Frame builders shared by the viewer tests.</summary>
internal static class TestFrames
{
    public static VideoFrame Frame(int index) => new() { Rgb = new byte[2 * 2 * 3], Width = 2, Height = 2, Index = index };

    /// <summary>Yields <paramref name="count"/> frames from <paramref name="first"/>, each after a yield to the scheduler.</summary>
    public static async IAsyncEnumerable<VideoFrame> Run(int first, int count, [EnumeratorCancellation] CancellationToken cancel = default)
    {
        for (int i = 0; i < count; i++)
        {
            cancel.ThrowIfCancellationRequested();
            await Task.Yield();
            yield return Frame(first + i);
        }
    }
}
