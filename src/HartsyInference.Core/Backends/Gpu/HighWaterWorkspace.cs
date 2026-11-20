using HartsyInference.Core.Exceptions;

namespace HartsyInference.Core.Backends.Gpu;

/// <summary>One device scratch buffer a backend keeps at its high-water size instead of allocating per call, with leases
/// for the captured graphs that baked its address.</summary>
/// <remarks>
/// A captured graph replays with the pointers it recorded, so a buffer a live graph used must outlive that graph. A graph
/// that uses the buffer during its capture takes a lease (<see cref="Use"/> with a lessee); growth and release then retire
/// a leased buffer instead of freeing it, and it is freed when its last lessee calls <see cref="Release"/>. A buffer no
/// graph leases is freed before its replacement is allocated, so growth near the memory limit needs only the new size.
/// <para>A lease holds its lessee weakly, so it does not keep an abandoned graph alive. A lessee that was collected
/// without releasing can never replay, and its lease lapses the next time the workspace grows or releases; the buffer is
/// freed then, under the owner's lock, never from a finalizer.</para>
/// The owner serializes every call (a backend's op lock); only <see cref="Bytes"/> may be read without it.
/// </remarks>
internal sealed class HighWaterWorkspace(Func<long, ulong> allocate, Action<ulong> free)
{
    private sealed class Block(ulong pointer, long bytes)
    {
        public readonly ulong Pointer = pointer;
        public readonly long Bytes = bytes;
        private readonly List<WeakReference<object>> _lessees = [];

        /// <summary>Live lessees, after dropping the ones that were collected.</summary>
        public int Leases
        {
            get
            {
                _lessees.RemoveAll(static lease => !lease.TryGetTarget(out _));
                return _lessees.Count;
            }
        }

        public void Lease(object lessee)
        {
            if (IndexOf(lessee) >= 0) return;
            // Drop collected lessees first, so graphs captured and abandoned without a growth cannot pile up here.
            _lessees.RemoveAll(static lease => !lease.TryGetTarget(out _));
            _lessees.Add(new WeakReference<object>(lessee));
        }

        public void Release(object lessee)
        {
            int index = IndexOf(lessee);
            if (index >= 0) _lessees.RemoveAt(index);
        }

        private int IndexOf(object lessee)
        {
            for (int i = 0; i < _lessees.Count; i++)
                if (_lessees[i].TryGetTarget(out object? held) && ReferenceEquals(held, lessee)) return i;
            return -1;
        }
    }

    private readonly List<Block> _retired = [];
    private Block? _current;
    private long _bytes;

    /// <summary>Bytes the current buffer holds; 0 when there is none. Safe to read without the owner's lock, as advice.</summary>
    public long Bytes => Volatile.Read(ref _bytes);

    /// <summary>The smallest size that did not fit since <see cref="ReleaseCurrent"/> last ran; such a size is not retried.</summary>
    public long GrowthFailedAt { get; internal set; } = long.MaxValue;

    /// <summary>Bytes of retired buffers kept for a lessee. A buffer whose lessee was collected still counts until the
    /// workspace next grows or releases.</summary>
    public long RetainedBytes
    {
        get
        {
            long total = 0;
            foreach (Block block in _retired) total += block.Bytes;
            return total;
        }
    }

    /// <summary>Live graphs that lease the current buffer.</summary>
    public int CurrentLeases => _current?.Leases ?? 0;

    /// <summary>The current buffer when it holds at least <paramref name="bytes"/>, else 0. A non-null
    /// <paramref name="lessee"/> (the graph being captured) takes a lease on the buffer it is handed.</summary>
    public ulong Use(long bytes, object? lessee)
    {
        Block? current = _current;
        if (current is null || bytes > current.Bytes) return 0;
        if (lessee is not null) current.Lease(lessee);
        return current.Pointer;
    }

    /// <summary>Replaces the buffer with one of <paramref name="bytes"/> and returns it, or 0 when that size is known not
    /// to fit. An unleased buffer is freed first; a leased one is retired, and stays current if its replacement does not fit.</summary>
    /// <exception cref="OutOfVramException">The allocation did not fit; the size is remembered and not retried.</exception>
    public ulong Grow(long bytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(bytes);
        if (bytes >= GrowthFailedAt) return 0;
        FreeUnleasedRetired();
        Block? old = _current;
        bool keepOld = old is { Leases: > 0 };
        if (old is not null && !keepOld)
        {
            // Free first: old and new together are what fails near the memory limit.
            SetCurrent(null);
            free(old.Pointer);
        }
        ulong pointer;
        try
        {
            pointer = allocate(bytes);
        }
        catch (OutOfVramException)
        {
            GrowthFailedAt = bytes;
            throw;
        }
        if (keepOld) _retired.Add(old!);
        SetCurrent(new Block(pointer, bytes));
        return pointer;
    }

    /// <summary>Drops every lease <paramref name="lessee"/> holds (its graph was disposed, reset or never finished
    /// capturing) and frees each retired buffer left without a lessee.</summary>
    public void Release(object lessee)
    {
        _current?.Release(lessee);
        foreach (Block block in _retired) block.Release(lessee);
        FreeUnleasedRetired();
    }

    /// <summary>Gives the current buffer back (activations are being freed) and lets a size that did not fit try again.
    /// A leased buffer is retired until its lessees release it.</summary>
    public void ReleaseCurrent()
    {
        GrowthFailedAt = long.MaxValue;
        Block? old = _current;
        if (old is not null)
        {
            SetCurrent(null);
            _retired.Add(old);
        }
        FreeUnleasedRetired();
    }

    /// <summary>Forgets every buffer, leased or not, and returns their pointers for the owner to free at teardown.</summary>
    public ulong[] Detach()
    {
        List<ulong> pointers = [];
        if (_current is not null) pointers.Add(_current.Pointer);
        foreach (Block block in _retired) pointers.Add(block.Pointer);
        _retired.Clear();
        SetCurrent(null);
        return [.. pointers];
    }

    private void FreeUnleasedRetired()
    {
        for (int i = _retired.Count - 1; i >= 0; i--)
        {
            Block block = _retired[i];
            if (block.Leases > 0) continue;
            _retired.RemoveAt(i);
            free(block.Pointer);
        }
    }

    private void SetCurrent(Block? block)
    {
        _current = block;
        Volatile.Write(ref _bytes, block?.Bytes ?? 0);
    }
}
