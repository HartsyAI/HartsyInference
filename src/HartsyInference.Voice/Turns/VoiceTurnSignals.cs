namespace HartsyInference.Voice.Turns;

/// <summary>The lock-free state the audio thread shares with the turn loop and the outbound reader: which turn is
/// audible, the outbound flush epoch, and the running turn's cancellation.</summary>
/// <remarks>Every field is a single word read and written with <see cref="Volatile"/> or <see cref="Interlocked"/>, so
/// the audio thread never waits on a thread that could be descheduled holding a lock. A barge-in is one compare
/// and swap on <see cref="SpeakingTurn"/>: whichever of the audio thread (barge-in) and the turn loop (reply finished)
/// clears it first owns the transition, so a reply that finished playing cannot be flushed and a barge-in cannot be
/// lost to a finishing turn.</remarks>
internal sealed class VoiceTurnSignals
{
    private int _speakingTurn;
    private long _flushEpoch;
    private VoiceTurnHandle? _turn;
    private int _bargedInTurn;
    private long _bargeInNs;

    /// <summary>The turn whose reply is audible, or 0.</summary>
    public int SpeakingTurn => Volatile.Read(ref _speakingTurn);

    /// <summary>Bumped for every flush; the outbound reader discards its queue when it sees a new value.</summary>
    public long FlushEpoch => Volatile.Read(ref _flushEpoch);

    /// <summary>The last turn the caller barged in on, or 0.</summary>
    public int BargedInTurn => Volatile.Read(ref _bargedInTurn);

    /// <summary>When that barge-in was decided, in monotonic nanoseconds.</summary>
    public long BargeInNs => Volatile.Read(ref _bargeInNs);

    /// <summary>Publishes the turn that is starting.</summary>
    public void BeginTurn(VoiceTurnHandle handle) => Volatile.Write(ref _turn, handle);

    /// <summary>Withdraws <paramref name="handle"/> and its audibility, unless a later turn replaced them.</summary>
    public void EndTurn(VoiceTurnHandle handle)
    {
        Interlocked.CompareExchange(ref _turn, null, handle);
        Interlocked.CompareExchange(ref _speakingTurn, 0, handle.TurnId);
    }

    /// <summary>Marks <paramref name="turnId"/> audible; barge-in is armed from here.</summary>
    public void BeginSpeaking(int turnId) => Volatile.Write(ref _speakingTurn, turnId);

    /// <summary>Clears audibility when the reply finished playing; false when a barge-in got there first.</summary>
    public bool EndSpeaking(int turnId) => Interlocked.CompareExchange(ref _speakingTurn, 0, turnId) == turnId;

    /// <summary>Asks the outbound reader to drop everything queued.</summary>
    public long RequestFlush() => Interlocked.Increment(ref _flushEpoch);

    /// <summary>Audio thread: stops <paramref name="turnId"/> if it is still audible. Flushes the outbound queue and
    /// cancels the turn without running any cancellation callback on the calling thread.</summary>
    /// <returns>False when the turn had already stopped speaking.</returns>
    public bool TryBargeIn(int turnId, long detectNs)
    {
        if (turnId == 0 || Interlocked.CompareExchange(ref _speakingTurn, 0, turnId) != turnId)
        {
            return false;
        }
        Interlocked.Increment(ref _flushEpoch);
        Volatile.Write(ref _bargeInNs, detectNs);
        Volatile.Write(ref _bargedInTurn, turnId);
        if (Volatile.Read(ref _turn) is { } handle && handle.TurnId == turnId)
        {
            // CancelAsync flips the token at once and runs the registered callbacks on the thread pool, so the audio
            // thread never executes the turn's cleanup; the task only reports those callbacks.
            _ = handle.Cancellation.CancelAsync();
        }
        return true;
    }
}
