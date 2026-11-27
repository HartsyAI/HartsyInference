namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>An open-addressing map from a packed voxel key to a dense index, growing as keys are added. Used by <see cref="NarrowBandRemesher"/> for the active voxels and the grid corners they touch; single writer, any number of readers once it stops growing.</summary>
internal sealed class VoxelKeyMap
{
    private const long Empty = -1;
    private long[] _keys;
    private int[] _values;
    private int _mask;

    public VoxelKeyMap(int capacity)
    {
        int size = 16;
        while (size < (long)capacity * 2)
            size <<= 1;
        _keys = new long[size];
        _values = new int[size];
        Array.Fill(_keys, Empty);
        _mask = size - 1;
    }

    /// <summary>The number of keys held.</summary>
    public int Count { get; private set; }

    /// <summary>The index of <paramref name="key"/>, adding it with the next index when it is new.</summary>
    public int GetOrAdd(long key, out bool added)
    {
        if ((long)(Count + 1) * 2 > _keys.Length)
            Grow();
        int slot = Slot(key);
        while (_keys[slot] != Empty)
        {
            if (_keys[slot] == key)
            {
                added = false;
                return _values[slot];
            }

            slot = (slot + 1) & _mask;
        }

        _keys[slot] = key;
        _values[slot] = Count;
        added = true;
        return Count++;
    }

    /// <summary>The index of <paramref name="key"/>, or -1.</summary>
    public int Find(long key)
    {
        int slot = Slot(key);
        while (_keys[slot] != Empty)
        {
            if (_keys[slot] == key)
                return _values[slot];
            slot = (slot + 1) & _mask;
        }

        return -1;
    }

    private int Slot(long key) => (int)(((ulong)key * 0x9E3779B97F4A7C15UL) >> 32) & _mask;

    private void Grow()
    {
        long[] oldKeys = _keys;
        int[] oldValues = _values;
        int size = oldKeys.Length * 2;
        _keys = new long[size];
        _values = new int[size];
        Array.Fill(_keys, Empty);
        _mask = size - 1;
        for (int i = 0; i < oldKeys.Length; i++)
        {
            if (oldKeys[i] == Empty)
                continue;
            int slot = Slot(oldKeys[i]);
            while (_keys[slot] != Empty)
                slot = (slot + 1) & _mask;
            _keys[slot] = oldKeys[i];
            _values[slot] = oldValues[i];
        }
    }
}
