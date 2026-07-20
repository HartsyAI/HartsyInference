using System.Runtime.CompilerServices;
using System.Threading.Channels;
using HartsyInference.Conditioning;
using HartsyInference.Core.Logging;
using HartsyInference.Video;

namespace HartsyInference.World.Sessions;

/// <summary>Default <see cref="IInteractiveSession"/>: one dedicated compute thread drives an <see cref="IFrameStepper"/>
/// from a bounded action queue into a bounded frame queue. Producers block when the action queue is full; consumers
/// falling behind cause the oldest unread frame to be dropped (latency over completeness). On action underrun the last
/// action is repeated so the world keeps streaming — interactive sessions never block the model on input.
///
/// <para>The session does NOT own the stepper or its model components (pipeline component-ownership convention) —
/// it only owns its queues and thread. Action payloads are copied into pooled buffers on submit, so the caller may
/// reuse its payload memory immediately.</para></summary>
public sealed class BackgroundComputeSession : IInteractiveSession
{
    private const int LatencyWindow = 256;

    private readonly IFrameStepper _stepper;
    private readonly Channel<ActionInput> _actions;
    private readonly Channel<VideoFrame> _frames;
    private readonly CancellationTokenSource _cts = new();
    private readonly Thread _thread;
    private readonly Queue<byte[]> _payloadPool = new();
    private readonly object _poolLock = new();
    private readonly int _maxPayloadBytes;
    private readonly double[] _stepMs = new double[LatencyWindow];
    private int _stepSamples;
    private long _framesEmitted;
    private long _droppedFrames;
    private long _currentFrameIndex = -1;
    private int _actionQueueDepth;
    private QualityProfile? _pendingProfile;
    private int _disposed;

    public BackgroundComputeSession(IFrameStepper stepper, int targetFps = 24,
        int actionQueueDepth = 8, int frameQueueDepth = 4, int maxPayloadBytes = 64)
    {
        ArgumentNullException.ThrowIfNull(stepper);
        if (targetFps <= 0) throw new ArgumentOutOfRangeException(nameof(targetFps));
        if (actionQueueDepth <= 0) throw new ArgumentOutOfRangeException(nameof(actionQueueDepth));
        if (frameQueueDepth <= 0) throw new ArgumentOutOfRangeException(nameof(frameQueueDepth));
        _stepper = stepper;
        TargetFps = targetFps;
        _maxPayloadBytes = maxPayloadBytes;
        _actions = Channel.CreateBounded<ActionInput>(new BoundedChannelOptions(actionQueueDepth)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
        });
        _frames = Channel.CreateBounded<VideoFrame>(new BoundedChannelOptions(frameQueueDepth)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleWriter = true,
        });
        _thread = new Thread(RunLoop) { IsBackground = true, Name = "HartsyInference.WorldSession" };
        _thread.Start();
    }

    public int FrameWidth => _stepper.FrameWidth;

    public int FrameHeight => _stepper.FrameHeight;

    public int TargetFps { get; }

    public long CurrentFrameIndex => Interlocked.Read(ref _currentFrameIndex);

    public async ValueTask SubmitActionAsync(ActionInput action, CancellationToken ct = default)
    {
        ThrowIfDisposed();
        if (action.Payload.Length > _maxPayloadBytes)
            throw new ArgumentException($"Action payload ({action.Payload.Length} B) exceeds the session's max of {_maxPayloadBytes} B.", nameof(action));
        byte[] buffer = RentPayloadBuffer();
        action.Payload.Span.CopyTo(buffer);
        ActionInput queued = action with { Payload = new ReadOnlyMemory<byte>(buffer, 0, action.Payload.Length) };
        await _actions.Writer.WriteAsync(queued, ct).ConfigureAwait(false);
        Interlocked.Increment(ref _actionQueueDepth);
    }

    public async IAsyncEnumerable<VideoFrame> ReadFramesAsync([EnumeratorCancellation] CancellationToken ct = default)
    {
        await foreach (VideoFrame frame in _frames.Reader.ReadAllAsync(ct).ConfigureAwait(false))
            yield return frame;
    }

    public InteractiveSessionStats GetStats()
    {
        int n = Math.Min(Volatile.Read(ref _stepSamples), LatencyWindow);
        double p50 = 0, p99 = 0;
        if (n > 0)
        {
            double[] window = new double[n];
            Array.Copy(_stepMs, window, n);
            Array.Sort(window);
            p50 = window[(int)((n - 1) * 0.50)];
            p99 = window[(int)((n - 1) * 0.99)];
        }
        return new InteractiveSessionStats(
            p50, p99,
            Interlocked.Read(ref _framesEmitted),
            Interlocked.Read(ref _droppedFrames),
            Volatile.Read(ref _actionQueueDepth),
            _frames.Reader.Count);
    }

    public void SetQualityProfile(QualityProfile profile)
    {
        ArgumentNullException.ThrowIfNull(profile);
        _pendingProfile = profile;   // applied on the compute thread at the next step boundary
    }

    private void RunLoop()
    {
        ActionInput? last = null;
        TimeSpan underrunWait = TimeSpan.FromMilliseconds(Math.Max(1.0, 500.0 / TargetFps));
        try
        {
            while (!_cts.IsCancellationRequested)
            {
                ActionInput action;
                bool fresh;
                if (_actions.Reader.TryRead(out ActionInput queued))
                {
                    Interlocked.Decrement(ref _actionQueueDepth);
                    action = queued;
                    fresh = true;
                }
                else if (last is null)
                {
                    // No action yet — block for the first one (or shutdown).
                    try
                    {
                        if (!_actions.Reader.WaitToReadAsync(_cts.Token).AsTask().GetAwaiter().GetResult()) break;
                    }
                    catch (OperationCanceledException) { break; }
                    continue;
                }
                else
                {
                    // Underrun: give the producer half a frame budget, then repeat the last action.
                    try
                    {
                        Task<bool> wait = _actions.Reader.WaitToReadAsync(_cts.Token).AsTask();
                        if (!wait.Wait(underrunWait)) { action = RepeatLast(last.Value); fresh = false; }
                        else if (wait.Result && _actions.Reader.TryRead(out queued)) { Interlocked.Decrement(ref _actionQueueDepth); action = queued; fresh = true; }
                        else break;   // channel completed
                    }
                    catch (AggregateException) { break; }
                    catch (OperationCanceledException) { break; }
                }

                QualityProfile? profile = Interlocked.Exchange(ref _pendingProfile, null);
                if (profile is not null) _stepper.ApplyQualityProfile(profile);

                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                IReadOnlyList<VideoFrame> produced = _stepper.Step(action);
                double ms = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                int slot = _stepSamples % LatencyWindow;
                _stepMs[slot] = ms;
                Volatile.Write(ref _stepSamples, _stepSamples + 1);

                if (fresh)
                {
                    // Keep this action's pooled payload as the repeat source; recycle the previous one.
                    if (last is not null) ReturnPayloadBuffer(last.Value);
                    last = action;
                }
                else
                {
                    last = action;   // same buffer, advanced frame index
                }

                for (int i = 0; i < produced.Count; i++)
                    EmitFrame(produced[i]);
            }
        }
        catch (Exception ex)
        {
            Logs.Error("Interactive session compute thread failed", ex);
            _frames.Writer.TryComplete(ex);
            return;
        }
        _frames.Writer.TryComplete();
    }

    private ActionInput RepeatLast(in ActionInput last) =>
        last with { FrameIndex = last.FrameIndex + 1 };

    private void EmitFrame(VideoFrame frame)
    {
        if (!_frames.Writer.TryWrite(frame))
        {
            // Consumer is behind: drop the oldest unread frame to bound latency.
            if (_frames.Reader.TryRead(out _)) Interlocked.Increment(ref _droppedFrames);
            _frames.Writer.TryWrite(frame);
        }
        Interlocked.Increment(ref _framesEmitted);
        Interlocked.Exchange(ref _currentFrameIndex, frame.Index);
    }

    private byte[] RentPayloadBuffer()
    {
        lock (_poolLock)
        {
            if (_payloadPool.Count > 0) return _payloadPool.Dequeue();
        }
        return new byte[_maxPayloadBytes];
    }

    private void ReturnPayloadBuffer(in ActionInput action)
    {
        if (!System.Runtime.InteropServices.MemoryMarshal.TryGetArray(action.Payload, out ArraySegment<byte> segment) || segment.Array is null)
            return;
        lock (_poolLock)
        {
            if (_payloadPool.Count < 32) _payloadPool.Enqueue(segment.Array);
        }
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(BackgroundComputeSession));
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _actions.Writer.TryComplete();
        _cts.Cancel();
        await Task.Run(_thread.Join).ConfigureAwait(false);
        _cts.Dispose();
    }
}
