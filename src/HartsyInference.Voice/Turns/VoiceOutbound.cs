using HartsyInference.Core.Runtime;

namespace HartsyInference.Voice.Turns;

/// <summary>The reply audio queue between the turn loop (producer) and whoever plays it (consumer, through
/// <see cref="VoiceAgentSession.ReadOutbound"/>): an <see cref="SpscRing{T}"/> plus the flush protocol a barge-in needs.</summary>
/// <remarks>The ring allows <see cref="SpscRing{T}.DiscardAll"/> only on the consumer, so a flush is a request: the
/// audio thread bumps <see cref="VoiceTurnSignals.FlushEpoch"/> and the reader discards on its next read, then
/// publishes the epoch it applied. Two rules keep a flush from eating the wrong audio. The producer re-checks the
/// epoch after every write and, when it moved, bumps it again, so a frame that landed after the reader's discard is
/// discarded too. And the producer never starts a turn while the reader has not applied the latest epoch, so a
/// pending discard can never take the next reply's opening. The producer waits for space, playback or an applied flush
/// on a waiter that carries what it waits for; the reader completes it once, on the read that reaches it, not on every
/// read. The reader never blocks and never allocates: the waiter's only continuation is the producer's own, so
/// completing it queues that continuation and nothing else, and cancellation completes the waiter from the token's
/// side.</remarks>
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
    private long _wakes;
    private Waiter? _waiter;

    public VoiceOutbound(int capacity, VoiceTurnSignals signals)
    {
        ArgumentNullException.ThrowIfNull(signals);
        _ring = new SpscRing<float>(capacity);
        _signals = signals;
        RefillSamples = Math.Max(1, _ring.Capacity / 4);
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

    /// <summary>Free space a blocked writer waits for before it writes again: its whole remaining write when that is
    /// smaller, otherwise a quarter of the ring, so a full queue costs the reader one wake per quarter rather than one
    /// per read.</summary>
    internal int RefillSamples { get; }

    /// <summary>Waits the reader has completed: one per producer wait whose position, space or epoch a read reached.</summary>
    internal long Wakes => Volatile.Read(ref _wakes);

    /// <summary>Whether the producer is parked on a wait the reader has not completed yet.</summary>
    internal bool ProducerWaiting => Volatile.Read(ref _waiter) is not null;

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
            int wanted = Math.Min(count, RefillSamples);
            if (free < wanted)
            {
                await WaitAsync(WaitFor.Space, wanted, cancel).ConfigureAwait(false);
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
            cancel.ThrowIfCancellationRequested();
            Waiter waiter = new(condition, target);
            // Exchange is a full fence: the re-check below cannot be ordered before the waiter is visible to the reader.
            Interlocked.Exchange(ref _waiter, waiter);
            if (Satisfied(condition, target))
            {
                Interlocked.CompareExchange(ref _waiter, null, waiter);
                return;
            }
            // The token completes the waiter itself. Awaiting waiter.Task.WaitAsync(cancel) would hang a cancellation
            // promise on the task, and the reader's completion would queue it through a new 32-byte invoker on the
            // reader's thread; with this method's continuation the only one, completing the waiter allocates nothing there.
            CancellationTokenRegistration registration = cancel.UnsafeRegister(
                static (state, token) => ((Waiter)state!).TrySetCanceled(token), waiter);
            try
            {
                await waiter.Task.ConfigureAwait(false);
            }
            finally
            {
                registration.Dispose();
                // A cancelled waiter is withdrawn, so the reader never completes it later.
                Interlocked.CompareExchange(ref _waiter, null, waiter);
            }
        }
    }

    private void WakeProducer()
    {
        // Pairs with the fence in WaitAsync: the positions published above are visible before the waiter is read.
        Interlocked.MemoryBarrier();
        Waiter? waiter = Volatile.Read(ref _waiter);
        if (waiter is null || !Satisfied(waiter.Condition, waiter.Target))
        {
            return;
        }
        if (Interlocked.CompareExchange(ref _waiter, null, waiter) == waiter && waiter.TrySetResult())
        {
            Volatile.Write(ref _wakes, _wakes + 1);
        }
    }

    /// <summary>One producer wait: what it waits for, completed by the reader on the read that reaches it.</summary>
    private sealed class Waiter(WaitFor condition, long target) : TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously)
    {
        public WaitFor Condition { get; } = condition;

        public long Target { get; } = target;
    }
}
