namespace HartsyInference.Core.Runtime;

/// <summary>Lock-free ring for exactly one producer thread and one consumer thread.</summary>
/// <remarks>Built for the seam between an ordinary thread that fills it and a real-time thread that drains it: a
/// <c>Monitor</c> there is a priority inversion, because a descheduled producer holding the lock stalls a consumer
/// that outranks it. Here the two sides share only two <see cref="Volatile"/> indices — the producer publishes its
/// write index after the copy, the consumer publishes its read index after its own — so neither ever waits on the
/// other. Nothing is allocated after construction.
/// <para>A full ring drops the newest data: <see cref="Write"/> stores what fits and counts the rest in
/// <see cref="DroppedSamples"/>, so a burst from the producer can never overwrite samples the consumer has not yet
/// played. Any other thread may read the counters, but only the producer may call <see cref="Write"/> and only the
/// consumer <see cref="Read"/> and <see cref="DiscardAll"/>.</para></remarks>
/// <typeparam name="T">Element type; audio callers use <see cref="short"/> or <see cref="float"/>.</typeparam>
public sealed class SpscRing<T>
{
    private readonly T[] _buffer;
    private readonly int _mask;
    private long _head;
    private long _tail;
    private long _dropped;

    /// <summary>Creates a ring holding <paramref name="capacity"/> elements, which must be a power of two.</summary>
    public SpscRing(int capacity)
    {
        if (capacity <= 0 || (capacity & (capacity - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "capacity must be a positive power of two");
        }
        _buffer = new T[capacity];
        _mask = capacity - 1;
    }

    /// <summary>Elements the ring can hold.</summary>
    public int Capacity => _buffer.Length;

    /// <summary>Elements written and not yet read.</summary>
    public int Available => (int)(Volatile.Read(ref _tail) - Volatile.Read(ref _head));

    /// <summary>Elements <see cref="Write"/> can accept before dropping.</summary>
    public int FreeSpace => _buffer.Length - Available;

    /// <summary>Elements refused by <see cref="Write"/> because the ring was full.</summary>
    public long DroppedSamples => Volatile.Read(ref _dropped);

    /// <summary>Appends as much of <paramref name="source"/> as fits; the remainder is dropped and counted.</summary>
    /// <returns>The number of elements stored.</returns>
    public int Write(ReadOnlySpan<T> source)
    {
        long tail = _tail;
        long head = Volatile.Read(ref _head);
        int free = _buffer.Length - (int)(tail - head);
        int count = Math.Min(source.Length, free);
        if (count > 0)
        {
            int index = (int)(tail & _mask);
            int first = Math.Min(count, _buffer.Length - index);
            source[..first].CopyTo(_buffer.AsSpan(index, first));
            if (count > first)
            {
                source.Slice(first, count - first).CopyTo(_buffer.AsSpan(0, count - first));
            }
            Volatile.Write(ref _tail, tail + count);
        }
        int dropped = source.Length - count;
        if (dropped > 0)
        {
            Volatile.Write(ref _dropped, _dropped + dropped);
        }
        return count;
    }

    /// <summary>Moves up to <c>destination.Length</c> elements out of the ring, oldest first.</summary>
    /// <returns>The number of elements copied; zero when the ring is empty.</returns>
    public int Read(Span<T> destination)
    {
        long head = _head;
        long tail = Volatile.Read(ref _tail);
        int count = Math.Min(destination.Length, (int)(tail - head));
        if (count <= 0)
        {
            return 0;
        }
        int index = (int)(head & _mask);
        int first = Math.Min(count, _buffer.Length - index);
        _buffer.AsSpan(index, first).CopyTo(destination);
        if (count > first)
        {
            _buffer.AsSpan(0, count - first).CopyTo(destination[first..]);
        }
        Volatile.Write(ref _head, head + count);
        return count;
    }

    /// <summary>Consumer-side flush: throws away everything written so far, including data the producer may be publishing concurrently.</summary>
    /// <returns>The number of elements discarded.</returns>
    public int DiscardAll()
    {
        long tail = Volatile.Read(ref _tail);
        long head = _head;
        int count = (int)(tail - head);
        if (count > 0)
        {
            Volatile.Write(ref _head, tail);
        }
        return count;
    }
}
