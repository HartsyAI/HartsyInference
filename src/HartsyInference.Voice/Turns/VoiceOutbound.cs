using System.Diagnostics;
using HartsyInference.Core.Runtime;

namespace HartsyInference.Voice.Turns;

/// <summary>The reply audio queue between the turn loop (producer) and whoever plays it (consumer, through
/// <see cref="VoiceAgentSession.ReadOutbound(Span{float}, out int)"/>): an <see cref="SpscRing{T}"/> plus the flush protocol
/// a barge-in needs and the turn each sample belongs to.</summary>
/// <remarks>The ring allows <see cref="SpscRing{T}.DiscardAll"/> only on the consumer, so a flush is a request: the
/// audio thread bumps <see cref="VoiceTurnSignals.FlushEpoch"/> and the reader discards on its next read, then
/// publishes the epoch it applied. Two rules keep a flush from eating the wrong audio. The producer re-checks the
/// epoch after every write and, when it moved, bumps it again, so a frame that landed after the reader's discard is
/// discarded too. And the producer never starts a turn while the reader has not applied the latest epoch, so a
/// pending discard can never take the next reply's opening. The producer waits for space, playback, an applied flush
/// or room for a turn mark on a waiter that carries what it waits for; the reader completes it once, on the read that
/// reaches it, not on every read. The reader never blocks and never allocates: the waiter's only continuation is the
/// producer's own, so completing it queues that continuation and nothing else, and cancellation completes the waiter
/// from the token's side. The allocation a wait does need (its waiter, its registration and, once it suspends, the
/// async state machine) is the producer's, on the turn loop. There is one producer, so at most one wait is ever armed.
/// The one exception on the reader is the runtime's: the first continuation a thread ever queues to the pool allocates
/// once on that thread (32 B, or 192 B on a freshly started thread), and none after that.
/// <para>Turn marks: a turn publishes its id and the write position of its first sample before writing it, so a
/// reader that sees a sample also sees the mark that covers it, as long as it takes the ring's fill level before it
/// reads the marks. A tagged read stops at the next mark, so it never returns two turns' audio as one, and a remote
/// player can drop what a flush superseded by turn id alone.</para></remarks>
internal sealed class VoiceOutbound
{
    private const int TurnMarkCapacity = 16;

    private readonly SpscRing<float> _ring;
    private readonly VoiceTurnSignals _signals;
    private readonly long[] _markPositions = new long[TurnMarkCapacity];
    private readonly int[] _markTurns = new int[TurnMarkCapacity];
    private long _readerEpoch;
    private long _readerPosition;
    private long _consumed;
    private long _appliedEpoch;
    private long _written;
    private long _discarded;
    private long _lastDiscardNs;
    private long _marksWritten;
    private long _marksRead;
    private int _readTurn;
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
        MarkSpace,
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
    /// <paramref name="destination"/> and returns how many samples were real, reading across turn boundaries. One
    /// thread only; never blocks.</summary>
    public int Read(Span<float> destination)
    {
        int total = 0;
        while (total < destination.Length)
        {
            int read = Read(destination[total..], out _);
            if (read == 0)
            {
                break;
            }
            total += read;
        }
        return total;
    }

    /// <summary>Reader: like <see cref="Read(Span{float})"/>, but returns the samples of one turn only, stopping at the
    /// next turn's first sample. <paramref name="turnId"/> is the turn that wrote them; 0 when nothing was read, or for
    /// audio written without a turn mark.</summary>
    public int Read(Span<float> destination, out int turnId)
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
        // Fill level first, marks second: a mark is published before its turn's first sample, so every sample
        // counted here is covered by a mark read below.
        int limit = Math.Min(destination.Length, _ring.Available);
        long marks = Volatile.Read(ref _marksWritten);
        while (_marksRead < marks && _markPositions[Slot(_marksRead)] <= _readerPosition)
        {
            _readTurn = _markTurns[Slot(_marksRead)];
            Volatile.Write(ref _marksRead, _marksRead + 1);
            progressed = true;
        }
        if (_marksRead < marks)
        {
            limit = (int)Math.Min(limit, _markPositions[Slot(_marksRead)] - _readerPosition);
        }
        int read = limit > 0 ? _ring.Read(destination[..limit]) : 0;
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
        turnId = read > 0 ? _readTurn : 0;
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

    /// <summary>Producer: <see cref="WaitFlushesAppliedAsync"/>, then marks every sample written from here on as
    /// <paramref name="turnId"/>'s. Returns the epoch the turn writes under.</summary>
    public async ValueTask<long> BeginTurnAsync(int turnId, CancellationToken cancel)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(turnId);
        long epoch = await WaitFlushesAppliedAsync(cancel).ConfigureAwait(false);
        await WaitAsync(WaitFor.MarkSpace, 0, cancel).ConfigureAwait(false);
        long mark = _marksWritten;
        _markPositions[Slot(mark)] = _written;
        _markTurns[Slot(mark)] = turnId;
        Volatile.Write(ref _marksWritten, mark + 1);
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

    private static int Slot(long mark) => (int)(mark & (TurnMarkCapacity - 1));

    private bool Satisfied(WaitFor condition, long target) => condition switch
    {
        WaitFor.Space => _ring.FreeSpace >= target,
        WaitFor.Played => Volatile.Read(ref _consumed) >= target,
        WaitFor.MarkSpace => _marksWritten - Volatile.Read(ref _marksRead) < TurnMarkCapacity,
        _ => Volatile.Read(ref _appliedEpoch) >= target,
    };

    private async ValueTask WaitAsync(WaitFor condition, long target, CancellationToken cancel)
    {
        while (!Satisfied(condition, target))
        {
            cancel.ThrowIfCancellationRequested();
            Waiter waiter = new(condition, target);
            // Exchange is a full fence: the re-check below cannot be ordered before the waiter is visible to the reader.
            Waiter? armed = Interlocked.Exchange(ref _waiter, waiter);
            Debug.Assert(armed is null, "A second producer wait overlapped the first; the queue has a single producer.");
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
