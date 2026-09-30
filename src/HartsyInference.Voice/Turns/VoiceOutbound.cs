using HartsyInference.Core.Runtime;

namespace HartsyInference.Voice.Turns;

/// <summary>The reply audio queue between the turn loop (producer) and whoever plays it (consumer, through
/// <see cref="VoiceAgentSession.ReadOutbound"/>): an <see cref="SpscRing{T}"/> plus the flush protocol a barge-in needs.</summary>
/// <remarks>The ring allows <see cref="SpscRing{T}.DiscardAll"/> only on the consumer, so a flush is a request: the
/// audio thread bumps <see cref="VoiceTurnSignals.FlushEpoch"/> and the reader discards on its next read, then
/// publishes the epoch it applied. Two rules keep a flush from eating the wrong audio. The producer re-checks the
/// epoch after every write and, when it moved, bumps it again, so a frame that landed after the reader's discard is
/// discarded too. And the producer never starts a turn while the reader has not applied the latest epoch, so a
/// pending discard can never take the next reply's opening. The producer waits (space, playback, an applied flush)
/// on one completion the reader signals after every read that made progress; the reader never blocks and never
/// allocates.</remarks>
internal sealed class VoiceOutbound
{
    private readonly SpscRing<float> _ring;
    private readonly VoiceTurnSignals _signals;
    private long _readerEpoch;
    private long _readerPosition;
    private long _consumed;
    private long _appliedEpoch;
    private long _written;
    private long _discarded;
    private long _lastDiscardNs;
    private TaskCompletionSource? _waiter;

    public VoiceOutbound(int capacity, VoiceTurnSignals signals)
    {
        ArgumentNullException.ThrowIfNull(signals);
        _ring = new SpscRing<float>(capacity);
        _signals = signals;
    }

    private enum WaitFor
    {
        Space,
        Played,
        FlushApplied,
    }

    /// <summary>Samples the queue holds.</summary>
    public int Capacity => _ring.Capacity;

    /// <summary>Samples queued and not yet read or discarded.</summary>
    public int Queued => _ring.Available;

    /// <summary>Producer position: samples written since the session started.</summary>
    public long Written => Volatile.Read(ref _written);

    /// <summary>Reader position: samples read or discarded since the session started.</summary>
    public long Consumed => Volatile.Read(ref _consumed);

    /// <summary>Samples discarded by flushes.</summary>
    public long DiscardedSamples => Volatile.Read(ref _discarded);

    /// <summary>When the reader last applied a flush, in monotonic nanoseconds; 0 before the first.</summary>
    public long LastDiscardNs => Volatile.Read(ref _lastDiscardNs);

    /// <summary>Reader: applies a pending flush, copies queued samples, zero-fills the rest of
    /// <paramref name="destination"/> and returns how many samples were real. One thread only; never blocks.</summary>
    public int Read(Span<float> destination)
    {
        bool progressed = false;
        long epoch = _signals.FlushEpoch;
        if (epoch != _readerEpoch)
        {
            int dropped = _ring.DiscardAll();
            _readerEpoch = epoch;
            _readerPosition += dropped;
            Volatile.Write(ref _discarded, _discarded + dropped);
            Volatile.Write(ref _consumed, _readerPosition);
            Volatile.Write(ref _lastDiscardNs, MonotonicClock.NowNs());
            Volatile.Write(ref _appliedEpoch, epoch);
            progressed = true;
        }
        int read = _ring.Read(destination);
        if (read > 0)
        {
            _readerPosition += read;
            Volatile.Write(ref _consumed, _readerPosition);
            progressed = true;
        }
        destination[read..].Clear();
        if (progressed)
        {
            WakeProducer();
        }
        return read;
    }

    /// <summary>Producer: waits until the reader applied every flush requested so far, then returns that epoch, the
    /// one a new turn writes under.</summary>
    public async ValueTask<long> WaitFlushesAppliedAsync(CancellationToken cancel)
    {
        long epoch = _signals.FlushEpoch;
        await WaitAsync(WaitFor.FlushApplied, epoch, cancel).ConfigureAwait(false);
        return epoch;
    }

    /// <summary>Producer: queues <paramref name="count"/> samples, waiting for space rather than dropping any.</summary>
    /// <returns>False when a flush superseded <paramref name="epoch"/>; nothing more should be written for that turn.</returns>
    public async ValueTask<bool> WriteAsync(float[] samples, int offset, int count, long epoch, CancellationToken cancel)
    {
        while (count > 0)
        {
            cancel.ThrowIfCancellationRequested();
            if (_signals.FlushEpoch != epoch)
            {
                return false;
            }
            int free = _ring.FreeSpace;
            if (free == 0)
            {
                await WaitAsync(WaitFor.Space, 1, cancel).ConfigureAwait(false);
                continue;
            }
            int n = Math.Min(free, count);
            _ring.Write(samples.AsSpan(offset, n));
            Volatile.Write(ref _written, _written + n);
            offset += n;
            count -= n;
            if (_signals.FlushEpoch != epoch)
            {
                // The flush may have been applied before these samples landed; one more epoch makes the reader drop them.
                _signals.RequestFlush();
                return false;
            }
        }
        return true;
    }

    /// <summary>Producer: waits until the reader has consumed up to <paramref name="position"/>.</summary>
    public ValueTask WaitPlayedAsync(long position, CancellationToken cancel) => WaitAsync(WaitFor.Played, position, cancel);

    private bool Satisfied(WaitFor condition, long target) => condition switch
    {
        WaitFor.Space => _ring.FreeSpace >= target,
        WaitFor.Played => Volatile.Read(ref _consumed) >= target,
        _ => Volatile.Read(ref _appliedEpoch) >= target,
    };

    private async ValueTask WaitAsync(WaitFor condition, long target, CancellationToken cancel)
    {
        while (!Satisfied(condition, target))
        {
            TaskCompletionSource waiter = new(TaskCreationOptions.RunContinuationsAsynchronously);
            // Exchange is a full fence: the re-check below cannot be ordered before the waiter is visible to the reader.
            Interlocked.Exchange(ref _waiter, waiter);
            if (Satisfied(condition, target))
            {
                Interlocked.CompareExchange(ref _waiter, null, waiter);
                return;
            }
            await waiter.Task.WaitAsync(cancel).ConfigureAwait(false);
        }
    }

    private void WakeProducer()
    {
        // Pairs with the fence in WaitAsync: the positions published above are visible before the waiter is read.
        Interlocked.MemoryBarrier();
        TaskCompletionSource? waiter = Volatile.Read(ref _waiter);
        if (waiter is not null && Interlocked.CompareExchange(ref _waiter, null, waiter) == waiter)
        {
            waiter.TrySetResult();
        }
    }
}
