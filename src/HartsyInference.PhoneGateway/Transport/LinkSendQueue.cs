using HartsyInference.PhoneLink;

namespace HartsyInference.PhoneGateway.Transport;

/// <summary>The two send lanes in front of the link writer thread. Audio is a fixed ring of 20 ms frames that drops its
/// oldest entry when full, so the pump thread never blocks on a slow socket; control is a bounded queue whose senders
/// wait for room up to a timeout, so a call event is not lost to a burst while the link is up (a lane still full after
/// the wait means the link is wedged, and the caller decides what to do). Every caller is an ordinary thread, so one
/// short lock serves both lanes.</summary>
internal sealed class LinkSendQueue
{
    private readonly object _lock = new();
    private readonly short[][] _audioPcm;
    private readonly uint[] _audioCallId;
    private readonly bool[] _audioConcealed;
    private readonly LinkControlItem[] _control;
    private readonly int _controlTimeoutMs;
    private int _audioHead;
    private int _audioCount;
    private int _controlHead;
    private int _controlCount;
    private long _audioDropped;
    private long _cleared;

    public LinkSendQueue(int audioDepth, int controlDepth, int controlTimeoutMs)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(audioDepth, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(controlDepth, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(controlTimeoutMs, 1);
        _audioPcm = new short[audioDepth][];
        for (int i = 0; i < audioDepth; i++)
        {
            _audioPcm[i] = new short[LinkProtocol.InboundFrameSamples];
        }
        _audioCallId = new uint[audioDepth];
        _audioConcealed = new bool[audioDepth];
        _control = new LinkControlItem[controlDepth];
        _controlTimeoutMs = controlTimeoutMs;
    }

    /// <summary>Audio frames pushed out by newer ones because the writer fell behind.</summary>
    public long AudioDropped => Volatile.Read(ref _audioDropped);

    /// <summary>Frames of both lanes thrown away by <see cref="Clear"/> at disconnect.</summary>
    public long Cleared => Volatile.Read(ref _cleared);

    public int AudioCount
    {
        get
        {
            lock (_lock)
            {
                return _audioCount;
            }
        }
    }

    public int ControlCount
    {
        get
        {
            lock (_lock)
            {
                return _controlCount;
            }
        }
    }

    /// <summary>Queues one inbound frame, dropping the oldest queued frame when the lane is full. Never blocks.</summary>
    public void EnqueueAudio(uint callId, ReadOnlySpan<short> pcm, bool concealed)
    {
        if (pcm.Length != LinkProtocol.InboundFrameSamples)
        {
            throw new ArgumentException($"Inbound frames carry {LinkProtocol.InboundFrameSamples} samples, got {pcm.Length}.", nameof(pcm));
        }
        lock (_lock)
        {
            int capacity = _audioPcm.Length;
            if (_audioCount == capacity)
            {
                _audioHead = (_audioHead + 1) % capacity;
                _audioCount--;
                _audioDropped++;
            }
            int slot = (_audioHead + _audioCount) % capacity;
            pcm.CopyTo(_audioPcm[slot]);
            _audioCallId[slot] = callId;
            _audioConcealed[slot] = concealed;
            _audioCount++;
            Monitor.Pulse(_lock);
        }
    }

    /// <summary>Queues a control frame, waiting up to the configured timeout for room when the lane is full.</summary>
    /// <returns>False when the lane stayed full for the configured timeout: the link is wedged and the frame was not
    /// queued.</returns>
    public bool TryEnqueueControl(in LinkControlItem item) => TryEnqueueControl(item, _controlTimeoutMs);

    /// <summary>Queues a control frame, waiting up to <paramref name="timeoutMs"/> for room when the lane is full; zero
    /// does not wait at all.</summary>
    /// <returns>False when the lane stayed full for <paramref name="timeoutMs"/>; the frame was not queued.</returns>
    public bool TryEnqueueControl(in LinkControlItem item, int timeoutMs)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(timeoutMs);
        lock (_lock)
        {
            int capacity = _control.Length;
            long deadline = Environment.TickCount64 + timeoutMs;
            while (_controlCount == capacity)
            {
                long remaining = deadline - Environment.TickCount64;
                if (remaining <= 0 || !Monitor.Wait(_lock, (int)remaining))
                {
                    return false;
                }
            }
            _control[(_controlHead + _controlCount) % capacity] = item;
            _controlCount++;
            Monitor.Pulse(_lock);
            return true;
        }
    }

    /// <summary>Waits up to <paramref name="timeoutMs"/> for an item and takes it, control before audio.</summary>
    /// <returns>False when nothing arrived in time.</returns>
    public bool TryDequeue(int timeoutMs, out LinkControlItem control, Span<short> audio, out uint audioCallId, out bool audioConcealed, out bool isAudio)
    {
        lock (_lock)
        {
            if (_controlCount == 0 && _audioCount == 0)
            {
                Monitor.Wait(_lock, timeoutMs);
            }
            if (_controlCount > 0)
            {
                control = _control[_controlHead];
                _control[_controlHead] = default;
                _controlHead = (_controlHead + 1) % _control.Length;
                _controlCount--;
                audioCallId = 0;
                audioConcealed = false;
                isAudio = false;
                Monitor.Pulse(_lock);
                return true;
            }
            if (_audioCount > 0)
            {
                _audioPcm[_audioHead].CopyTo(audio);
                audioCallId = _audioCallId[_audioHead];
                audioConcealed = _audioConcealed[_audioHead];
                _audioHead = (_audioHead + 1) % _audioPcm.Length;
                _audioCount--;
                control = default;
                isAudio = true;
                return true;
            }
            control = default;
            audioCallId = 0;
            audioConcealed = false;
            isAudio = false;
            return false;
        }
    }

    /// <summary>Drops both lanes: the frames belonged to a connection that is gone.</summary>
    public void Clear()
    {
        lock (_lock)
        {
            _cleared += _audioCount + _controlCount;
            _audioCount = 0;
            _audioHead = 0;
            Array.Clear(_control);
            _controlCount = 0;
            _controlHead = 0;
            Monitor.PulseAll(_lock);
        }
    }

    /// <summary>Wakes a writer blocked in <see cref="TryDequeue"/> so it can notice a stop request.</summary>
    public void Wake()
    {
        lock (_lock)
        {
            Monitor.PulseAll(_lock);
        }
    }
}
