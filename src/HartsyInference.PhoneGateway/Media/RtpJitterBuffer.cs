using HartsyInference.Audio.Io;

namespace HartsyInference.PhoneGateway.Media;

/// <summary>Fixed eight-slot (160 ms) reorder and concealment buffer for 20 ms G.711 frames.</summary>
/// <remarks><see cref="Push"/> runs on sipsorcery's receive thread and does one copy and a few index operations under
/// a short lock; <see cref="Pop"/> runs on the pump thread every 20 ms. Both threads are ordinary <c>SCHED_OTHER</c>
/// threads, so the lock is not a priority inversion. Playout starts once <see cref="TargetDepth"/> frames (60 ms)
/// are queued; a missing frame is concealed by repeating the previous one once and then by silence, and after
/// <see cref="MaxMisses"/> consecutive misses the stream is considered idle (silence suppression or the far end
/// gone) and the buffer waits for the next packet to pre-fill again. A timestamp that does not continue the
/// stream (a new talkspurt, an SSRC restart, or a packet more than the window away) resets the buffer to that
/// packet. When the far end's clock runs ahead of ours the depth creeps up; a depth two frames over target drops
/// the oldest frame and counts it in <see cref="Trimmed"/>.</remarks>
public sealed class RtpJitterBuffer
{
    /// <summary>Slots in the window.</summary>
    public const int Slots = 8;

    /// <summary>Bytes in one 20 ms G.711 frame.</summary>
    public const int FrameBytes = 160;

    /// <summary>Samples per frame at the 8 kHz RTP clock.</summary>
    public const int FrameTimestampUnits = 160;

    /// <summary>Frames queued before playout starts.</summary>
    public const int DefaultTargetDepth = 3;

    /// <summary>Consecutive missing frames after which the stream is treated as idle.</summary>
    public const int MaxMisses = 8;

    private const int SlotMask = Slots - 1;
    private const int TrimSlack = 2;
    private const long MaxTimestampJump = Slots * FrameTimestampUnits;

    private readonly object _lock = new();
    private readonly byte[] _data = new byte[Slots * FrameBytes];
    private readonly ushort[] _slotSeq = new ushort[Slots];
    private readonly byte[] _slotPayloadType = new byte[Slots];
    private readonly bool[] _occupied = new bool[Slots];
    private readonly byte[] _lastFrame = new byte[FrameBytes];
    private readonly int _targetDepth;
    private bool _active;
    private bool _prebuffering;
    private bool _haveLast;
    private byte _lastPayloadType;
    private ushort _nextSeq;
    private ushort _highestSeq;
    private uint _highestTimestamp;
    private int _depth;
    private int _misses;
    private long _received;
    private long _popped;
    private long _late;
    private long _lost;
    private long _duplicate;
    private long _reordered;
    private long _concealed;
    private long _resets;
    private long _trimmed;
    private long _badPayload;

    public RtpJitterBuffer(int targetDepth = DefaultTargetDepth)
    {
        if (targetDepth < 1 || targetDepth > Slots - TrimSlack)
        {
            throw new ArgumentOutOfRangeException(nameof(targetDepth), targetDepth, $"targetDepth must be 1..{Slots - TrimSlack}.");
        }
        _targetDepth = targetDepth;
    }

    /// <summary>Frames queued before playout starts.</summary>
    public int TargetDepth => _targetDepth;

    /// <summary>Frames currently queued ahead of the playout point.</summary>
    public int Depth
    {
        get
        {
            lock (_lock)
            {
                return _depth;
            }
        }
    }

    /// <summary>Queued audio in milliseconds.</summary>
    public int DepthMs => Depth * 20;

    /// <summary>True while a stream is established (packets are arriving and playout is running or pre-filling).</summary>
    public bool IsActive
    {
        get
        {
            lock (_lock)
            {
                return _active;
            }
        }
    }

    public long Received => Volatile.Read(ref _received);
    public long Popped => Volatile.Read(ref _popped);

    /// <summary>Packets that arrived after their playout slot had passed.</summary>
    public long Late => Volatile.Read(ref _late);

    /// <summary>Playout slots that had no packet when their time came.</summary>
    public long Lost => Volatile.Read(ref _lost);
    public long Duplicate => Volatile.Read(ref _duplicate);

    /// <summary>Packets that arrived behind a later sequence number but still in time.</summary>
    public long Reordered => Volatile.Read(ref _reordered);

    /// <summary>Pops answered by concealment.</summary>
    public long Concealed => Volatile.Read(ref _concealed);

    /// <summary>Times the buffer re-based on a packet that did not continue the stream.</summary>
    public long Resets => Volatile.Read(ref _resets);

    /// <summary>Frames dropped to pull an over-deep buffer back toward the target.</summary>
    public long Trimmed => Volatile.Read(ref _trimmed);

    /// <summary>Packets refused because the payload was not one 20 ms frame.</summary>
    public long BadPayload => Volatile.Read(ref _badPayload);

    /// <summary>Stores one packet. Called on the receive thread; copies the payload and returns.</summary>
    public void Push(ushort sequence, uint timestamp, bool marker, byte payloadType, ReadOnlySpan<byte> payload)
    {
        if (payload.Length != FrameBytes)
        {
            Interlocked.Increment(ref _badPayload);
            return;
        }
        lock (_lock)
        {
            _received++;
            if (!_active)
            {
                Rebase(sequence, timestamp, countReset: false);
            }
            else
            {
                int seqAhead = (short)(sequence - _highestSeq);
                long expected = (uint)(_highestTimestamp + (uint)(seqAhead * FrameTimestampUnits));
                long jump = Math.Abs((long)(int)(timestamp - (uint)expected));
                if ((marker && jump != 0) || jump > MaxTimestampJump)
                {
                    Rebase(sequence, timestamp, countReset: true);
                }
            }
            int delta = (short)(sequence - _nextSeq);
            if (delta < 0)
            {
                _late++;
                return;
            }
            if (delta >= Slots)
            {
                // Beyond the window: either seven-plus frames vanished or the pump is stalled. Start over from here.
                Rebase(sequence, timestamp, countReset: true);
                delta = 0;
            }
            int slot = sequence & SlotMask;
            if (_occupied[slot])
            {
                if (_slotSeq[slot] == sequence)
                {
                    _duplicate++;
                    return;
                }
                _depth--;
            }
            payload.CopyTo(_data.AsSpan(slot * FrameBytes, FrameBytes));
            _slotSeq[slot] = sequence;
            _slotPayloadType[slot] = payloadType;
            _occupied[slot] = true;
            _depth++;
            if ((short)(sequence - _highestSeq) < 0)
            {
                _reordered++;
            }
            else
            {
                _highestSeq = sequence;
                _highestTimestamp = timestamp;
            }
            if (_prebuffering && _depth >= _targetDepth)
            {
                _prebuffering = false;
            }
        }
    }

    /// <summary>Takes the next frame for playout. Called on the pump thread once per period.</summary>
    /// <param name="destination">Receives <see cref="FrameBytes"/> bytes for <see cref="JitterPopResult.Frame"/> and
    /// <see cref="JitterPopResult.Concealed"/>.</param>
    /// <param name="payloadType">The payload type of the frame or of the concealment's source.</param>
    public JitterPopResult Pop(Span<byte> destination, out byte payloadType)
    {
        if (destination.Length < FrameBytes)
        {
            throw new ArgumentException($"destination needs {FrameBytes} bytes, got {destination.Length}.", nameof(destination));
        }
        lock (_lock)
        {
            payloadType = _lastPayloadType;
            if (!_active || _prebuffering)
            {
                return JitterPopResult.Silence;
            }
            int slot = _nextSeq & SlotMask;
            if (_occupied[slot] && _slotSeq[slot] == _nextSeq)
            {
                Span<byte> frame = _data.AsSpan(slot * FrameBytes, FrameBytes);
                frame.CopyTo(destination);
                frame.CopyTo(_lastFrame);
                payloadType = _slotPayloadType[slot];
                _lastPayloadType = payloadType;
                _haveLast = true;
                _occupied[slot] = false;
                _depth--;
                _nextSeq++;
                _misses = 0;
                _popped++;
                TrimExcess();
                return JitterPopResult.Frame;
            }
            _nextSeq++;
            _lost++;
            _misses++;
            _concealed++;
            if (_misses > MaxMisses)
            {
                GoIdle();
                return JitterPopResult.Silence;
            }
            if (_misses == 1 && _haveLast)
            {
                _lastFrame.CopyTo(destination);
            }
            else
            {
                destination.Slice(0, FrameBytes).Fill(SilenceCode(_lastPayloadType));
            }
            return JitterPopResult.Concealed;
        }
    }

    /// <summary>Drops everything and waits for the next packet to start a fresh pre-fill.</summary>
    public void Reset()
    {
        lock (_lock)
        {
            GoIdle();
        }
    }

    private void Rebase(ushort sequence, uint timestamp, bool countReset)
    {
        Array.Clear(_occupied);
        _depth = 0;
        _misses = 0;
        _nextSeq = sequence;
        _highestSeq = sequence;
        _highestTimestamp = timestamp;
        _active = true;
        _prebuffering = true;
        _haveLast = false;
        if (countReset)
        {
            _resets++;
        }
    }

    private void GoIdle()
    {
        Array.Clear(_occupied);
        _depth = 0;
        _misses = 0;
        _active = false;
        _prebuffering = true;
        _haveLast = false;
    }

    /// <summary>Drops frames while the queue sits more than <see cref="TrimSlack"/> over target, oldest first.</summary>
    private void TrimExcess()
    {
        while (_depth > _targetDepth + TrimSlack)
        {
            int slot = _nextSeq & SlotMask;
            if (!_occupied[slot] || _slotSeq[slot] != _nextSeq)
            {
                return;
            }
            _occupied[slot] = false;
            _depth--;
            _nextSeq++;
            _trimmed++;
        }
    }

    private static byte SilenceCode(byte payloadType) =>
        payloadType == G711Formats.PcmaPayloadType ? G711.ALawSilence : G711.MuLawSilence;
}
