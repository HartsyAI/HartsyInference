namespace HartsyInference.Core.Engram;

/// <summary>Fixed-capacity LRU of packed rows. Slot bytes live in lazily allocated slabs so a large budget costs nothing until rows arrive. Not thread-safe.</summary>
internal sealed class EngramRowCache
{
    private const int SlotsPerSlabShift = 14;
    private const int SlotsPerSlab = 1 << SlotsPerSlabShift;
    private const int None = -1;

    private readonly int _packedRowBytes;
    private readonly Dictionary<long, int> _slotOfRow = new();
    private readonly List<byte[]> _slabs = new();
    private long[] _rowOfSlot = new long[64];
    private int[] _newer = new int[64];
    private int[] _older = new int[64];
    private int _used;
    private int _newest = None;
    private int _oldest = None;

    public EngramRowCache(int capacityRows, int packedRowBytes)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capacityRows, 1);
        Capacity = capacityRows;
        _packedRowBytes = packedRowBytes;
    }

    public int Capacity { get; }

    public int Count => _used;

    public long Evictions { get; private set; }

    public bool Contains(long row) => _slotOfRow.ContainsKey(row);

    /// <summary>Marks a resident row most recently used; false when it is not cached.</summary>
    public bool Touch(long row)
    {
        if (!_slotOfRow.TryGetValue(row, out int slot))
            return false;
        if (slot != _newest)
        {
            Unlink(slot);
            PushNewest(slot);
        }
        return true;
    }

    /// <summary>The packed bytes of a resident row, or empty when it is not cached. Does not change recency.</summary>
    public ReadOnlySpan<byte> Peek(long row) => _slotOfRow.TryGetValue(row, out int slot) ? SlotBytes(slot) : default;

    /// <summary>Claims a slot for <paramref name="row"/> (evicting the least recently used row when full) and returns its writable bytes.</summary>
    public Span<byte> Insert(long row)
    {
        int slot;
        if (_used < Capacity)
        {
            slot = _used++;
            GrowFor(slot);
        }
        else
        {
            slot = _oldest;
            _slotOfRow.Remove(_rowOfSlot[slot]);
            Unlink(slot);
            Evictions++;
        }
        _rowOfSlot[slot] = row;
        _slotOfRow[row] = slot;
        PushNewest(slot);
        return SlotBytes(slot);
    }

    private Span<byte> SlotBytes(int slot) =>
        _slabs[slot >> SlotsPerSlabShift].AsSpan((slot & (SlotsPerSlab - 1)) * _packedRowBytes, _packedRowBytes);

    private void GrowFor(int slot)
    {
        if (slot >= _rowOfSlot.Length)
        {
            int size = Math.Min(Math.Max(_rowOfSlot.Length * 2, slot + 1), Capacity);
            Array.Resize(ref _rowOfSlot, size);
            Array.Resize(ref _newer, size);
            Array.Resize(ref _older, size);
        }
        if ((slot >> SlotsPerSlabShift) >= _slabs.Count)
        {
            int slots = Math.Min(SlotsPerSlab, Capacity - (slot & ~(SlotsPerSlab - 1)));
            _slabs.Add(new byte[slots * _packedRowBytes]);
        }
    }

    private void Unlink(int slot)
    {
        int newer = _newer[slot], older = _older[slot];
        if (newer == None) _newest = older; else _older[newer] = older;
        if (older == None) _oldest = newer; else _newer[older] = newer;
    }

    private void PushNewest(int slot)
    {
        _newer[slot] = None;
        _older[slot] = _newest;
        if (_newest != None) _newer[_newest] = slot;
        _newest = slot;
        if (_oldest == None) _oldest = slot;
    }
}
