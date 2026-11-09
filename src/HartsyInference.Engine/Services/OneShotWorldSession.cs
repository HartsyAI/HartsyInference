using System.Runtime.CompilerServices;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Worlds;

namespace HartsyInference.Engine.Services;

/// <summary>An <see cref="IWorldSession"/> over a batch model. Actions are queued until the stream starts, then one run under the
/// engine's execution lease renders the whole clip, and its frames are delivered once that run completes. A session streams once:
/// a second enumeration throws, whether the first finished, was cancelled or failed.</summary>
internal sealed class OneShotWorldSession : IWorldSession
{
    private readonly IOneShotClipPipeline _pipeline;
    private readonly WorldPlan _plan;
    private readonly WorldExecutionLease _lease;
    private readonly WorldPhaseLedger? _ledger;
    private readonly object _gate = new();
    private readonly List<string> _actions = [];
    private bool _streamStarted;
    private bool _disposed;

    /// <summary>Creates a session that renders <paramref name="plan"/>'s clip with <paramref name="pipeline"/> under <paramref name="lease"/>.</summary>
    internal OneShotWorldSession(IOneShotClipPipeline pipeline, WorldPlan plan, WorldExecutionLease lease, WorldPhaseLedger? ledger)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(lease);
        _pipeline = pipeline;
        _plan = plan;
        _lease = lease;
        _ledger = ledger;
    }

    /// <inheritdoc/>
    public void SendAction(string action)
    {
        ArgumentNullException.ThrowIfNull(action);
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_streamStarted)
            {
                throw new InvalidOperationException(
                    $"'{_plan.ModelId}' is a batch model: it renders its whole clip in one run, so every action must be sent before StreamAsync starts.");
            }
            _actions.Add(action);
        }
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<VideoFrame> StreamAsync([EnumeratorCancellation] CancellationToken cancel)
    {
        string[] actions;
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_streamStarted)
            {
                throw new InvalidOperationException(
                    $"'{_plan.ModelId}' has already streamed its clip. Batch frames are delivered once; open a new session to generate again.");
            }
            _streamStarted = true;
            actions = _actions.ToArray();
        }
        IReadOnlyList<VideoFrame> frames = await RunAsync(actions, cancel).ConfigureAwait(false);
        for (int i = 0; i < frames.Count; i++)
        {
            // A cancel that lands during delivery drops the rest of the clip and ends the stream as cancelled.
            cancel.ThrowIfCancellationRequested();
            yield return frames[i];
        }
    }

    /// <summary>Runs the clip under the lease. The phase is released before the lease, so the device is free again only once the
    /// run's intermediates are gone.</summary>
    private async Task<IReadOnlyList<VideoFrame>> RunAsync(string[] actions, CancellationToken cancel)
    {
        using IDisposable lease = await _lease.AcquireAsync(cancel).ConfigureAwait(false);
        using WorldPhase phase = new WorldPhase($"rollout {_plan.ModelId}", _ledger);
        OneShotClipRequest request = new OneShotClipRequest { Plan = _plan, Actions = actions, Phase = phase };
        // Awaited to completion without the token: a cancelled clip must finish unwinding before the lease is released.
        Task<IReadOnlyList<VideoFrame>> run = Task.Run(() => _pipeline.GenerateClip(request, cancel));
        IReadOnlyList<VideoFrame> frames = await run.ConfigureAwait(false);
        // A run that completed while a cancel was pending is not a success: its output is discarded.
        cancel.ThrowIfCancellationRequested();
        ValidateClip(frames);
        return frames;
    }

    /// <summary>Rejects a clip the plan does not describe: a wrong frame count, a frame out of order, or pixels that do not fill the frame.</summary>
    private void ValidateClip(IReadOnlyList<VideoFrame> frames)
    {
        if (frames.Count != _plan.Frames)
        {
            throw new InvalidOperationException(
                $"'{_plan.ModelId}' returned {frames.Count} frames; its plan asks for {_plan.Frames}.");
        }
        for (int i = 0; i < frames.Count; i++)
        {
            VideoFrame frame = frames[i];
            if (frame.Index != i || frame.Width <= 0 || frame.Height <= 0
                || frame.Rgb.LongLength != (long)frame.Width * frame.Height * 3)
            {
                throw new InvalidOperationException($"'{_plan.ModelId}' returned a malformed frame at position {i}.");
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_gate)
        {
            _disposed = true;
            _actions.Clear();
        }
    }
}
