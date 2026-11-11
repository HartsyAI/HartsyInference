using System.Runtime.CompilerServices;
using System.Threading.Channels;
using HartsyInference.Core.Diagnostics;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Worlds;
using HartsyInference.World.Sessions;

namespace HartsyInference.Engine.Services;

/// <summary>An <see cref="IWorldSession"/> over a streaming model. One compute thread holds the engine's execution lease for
/// the whole stream: it opens the stream, generates chunk after chunk, and hands each chunk's frames to the consumer as soon
/// as the chunk is decoded. Actions sent at any time steer the chunks generated after them. The compute thread generates a
/// chunk only when the consumer has room for it (at most <c>maxBufferedChunks</c> finished chunks wait unread), so a chunk
/// reads the newest actions and a slow consumer holds the model back rather than piling frames up.</summary>
/// <remarks>A session streams once: a second enumeration throws. The stream ends after the plan's last frame, or, for an
/// unbounded plan, when the caller cancels or disposes the session. Disposing cancels the stream: a session waiting for the
/// lease never starts, a chunk in flight stops at its next denoising pass, undelivered frames are dropped, and the lease is
/// released only after the stream's memory is.</remarks>
internal sealed class ChunkedWorldSession : IWorldSession
{
    private readonly IChunkedWorldPipeline _pipeline;
    private readonly WorldPlan _plan;
    private readonly WorldExecutionLease _lease;
    private readonly WorldPhaseLedger? _ledger;
    private readonly int _maxBufferedChunks;
    private readonly object _gate = new();
    private readonly List<string> _pending = [];

    /// <summary>Cancelled by <see cref="Dispose"/>. Never disposed itself: the compute thread may still be unwinding with its
    /// token, and a source with no timer or wait handle holds nothing that needs releasing.</summary>
    private readonly CancellationTokenSource _closed = new();
    private IChunkedWorldStream? _stream;
    private bool _streamStarted;
    private bool _disposed;

    /// <summary>Creates a session that streams <paramref name="plan"/> with <paramref name="pipeline"/> under <paramref name="lease"/>.</summary>
    /// <param name="maxBufferedChunks">Finished chunks that may wait unread before the model pauses; at least 1.</param>
    internal ChunkedWorldSession(IChunkedWorldPipeline pipeline, WorldPlan plan, WorldExecutionLease lease, WorldPhaseLedger? ledger,
        int maxBufferedChunks = 1)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(lease);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxBufferedChunks, 1);
        _pipeline = pipeline;
        _plan = plan;
        _lease = lease;
        _ledger = ledger;
        _maxBufferedChunks = maxBufferedChunks;
    }

    /// <inheritdoc/>
    public bool DefersDeviceWork => true;

    /// <inheritdoc/>
    public void SendAction(string action)
    {
        ArgumentNullException.ThrowIfNull(action);
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_stream is null)
            {
                // Before the stream opens, and after it has ended: the action is kept for the open, or has nothing left to steer.
                _pending.Add(action);
                return;
            }
            _stream.SendAction(action);
        }
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<VideoFrame> StreamAsync([EnumeratorCancellation] CancellationToken cancel)
    {
        CancellationToken closed;
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_streamStarted)
            {
                throw new InvalidOperationException(
                    $"'{_plan.ModelId}' has already streamed. A streaming session runs once; open a new session to generate again.");
            }
            _streamStarted = true;
            closed = _closed.Token;
        }

        // The caller's cancel or the session's disposal, whichever comes first, stops the stream.
        using CancellationTokenSource linked = CancellationTokenSource.CreateLinkedTokenSource(cancel, closed);
        CancellationToken stop = linked.Token;
        Channel<VideoFrame[]> chunks = Channel.CreateBounded<VideoFrame[]>(
            new BoundedChannelOptions(_maxBufferedChunks) { SingleReader = true, SingleWriter = true });
        // One dedicated thread for the whole stream: a device context and the model's carried state stay on the thread that made them.
        Task producer = Task.Factory.StartNew(() => Produce(chunks.Writer, stop), CancellationToken.None,
            TaskCreationOptions.LongRunning, TaskScheduler.Default);
        try
        {
            await foreach (VideoFrame[] chunk in chunks.Reader.ReadAllAsync(stop).ConfigureAwait(false))
            {
                foreach (VideoFrame frame in chunk)
                {
                    // A cancel or disposal that lands during delivery drops the rest of the stream and ends it as cancelled.
                    stop.ThrowIfCancellationRequested();
                    yield return frame;
                }
            }
        }
        finally
        {
            // However the consumer left, the compute thread must finish unwinding before this returns: its failure has already
            // reached the consumer through the channel, so only the wait matters here.
            linked.Cancel();
            await producer.ConfigureAwait(false);
        }
    }

    /// <summary>The compute thread: holds the lease, opens the stream, and generates chunks until the plan is complete or the
    /// stream is stopped. The stream is released before the lease, so the device is free again only once the stream's memory
    /// is gone. Every way out completes the channel, with the failure when there is one.</summary>
    private void Produce(ChannelWriter<VideoFrame[]> writer, CancellationToken stop)
    {
        Exception? failure = null;
        try
        {
            using IDisposable lease = _lease.AcquireAsync(stop).GetAwaiter().GetResult();
            using WorldPhase phase = new WorldPhase($"stream {_plan.ModelId}", _ledger);
            IChunkedWorldStream stream = phase.Own(Open(phase, stop));
            try
            {
                Generate(stream, writer, stop);
            }
            finally
            {
                lock (_gate)
                {
                    _stream = null;
                }
            }
        }
        catch (OperationCanceledException ex)
        {
            failure = ex;
        }
        catch (Exception ex)
        {
            // The session boundary of the compute thread: the consumer rethrows this from its enumeration.
            Logs.Error($"[World] The '{_plan.ModelId}' stream failed.", ex);
            failure = ex;
        }
        finally
        {
            writer.TryComplete(failure);
        }
    }

    /// <summary>Opens the stream with the actions sent so far, then makes it the target of later ones. An action that arrives
    /// while the open is running is applied before the stream takes live actions, so none is lost or reordered.</summary>
    private IChunkedWorldStream Open(WorldPhase phase, CancellationToken stop)
    {
        string[] opening;
        lock (_gate)
        {
            opening = _pending.ToArray();
        }
        IChunkedWorldStream stream = _pipeline.OpenStream(new ChunkedWorldRequest { Plan = _plan, Actions = opening, Phase = phase }, stop);
        try
        {
            lock (_gate)
            {
                for (int i = opening.Length; i < _pending.Count; i++)
                {
                    stream.SendAction(_pending[i]);
                }
                _pending.Clear();
                _stream = stream;
            }
            return stream;
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    private void Generate(IChunkedWorldStream stream, ChannelWriter<VideoFrame[]> writer, CancellationToken stop)
    {
        int delivered = 0;
        while (!stream.Completed && (_plan.Unbounded || delivered < _plan.Frames))
        {
            // Room first, then the chunk: it is generated as late as the consumer allows, under the newest actions.
            long roomStart = WorldStageTimings.Start();
            bool room = writer.WaitToWriteAsync(stop).AsTask().GetAwaiter().GetResult();
            WorldStageTimings.Record(WorldStage.ChunkWaitForRoom, roomStart);
            if (!room)
            {
                return;
            }
            stop.ThrowIfCancellationRequested();
            int expected = stream.NextChunkFrames;
            IReadOnlyList<Video.VideoFrame> frames = stream.GenerateChunk(stop);
            VideoFrame[] chunk = ToFrames(frames, delivered, expected, stream);
            delivered += chunk.Length;
            if (!writer.TryWrite(chunk))
            {
                writer.WriteAsync(chunk, stop).AsTask().GetAwaiter().GetResult();
            }
        }
        if (!_plan.Unbounded && delivered != _plan.Frames)
        {
            throw new InvalidOperationException(
                $"'{_plan.ModelId}' streamed {delivered} frames; its plan asks for {_plan.Frames}.");
        }
    }

    /// <summary>Rejects a chunk the stream did not announce: a wrong frame count, a frame out of order, or pixels that do not
    /// fill the frame.</summary>
    private VideoFrame[] ToFrames(IReadOnlyList<Video.VideoFrame> frames, int delivered, int expected, IChunkedWorldStream stream)
    {
        if (frames.Count != expected || expected < 1)
        {
            throw new InvalidOperationException(
                $"'{_plan.ModelId}' returned a chunk of {frames.Count} frames after announcing {expected}.");
        }
        VideoFrame[] chunk = new VideoFrame[frames.Count];
        for (int i = 0; i < chunk.Length; i++)
        {
            Video.VideoFrame frame = frames[i];
            if (frame.Index != delivered + i || frame.Width != stream.FrameWidth || frame.Height != stream.FrameHeight
                || frame.Rgb is null || frame.Rgb.LongLength != (long)frame.Width * frame.Height * 3)
            {
                throw new InvalidOperationException($"'{_plan.ModelId}' returned a malformed frame at position {delivered + i}.");
            }
            chunk[i] = new VideoFrame { Rgb = frame.Rgb, Width = frame.Width, Height = frame.Height, Index = frame.Index };
        }
        return chunk;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            _pending.Clear();
        }
        // Outside the lock: cancellation runs the stream's callbacks synchronously. The compute thread still unwinds before
        // the lease is released.
        _closed.Cancel();
    }
}
