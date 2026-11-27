using System.Numerics;

namespace HartsyInference.ThreeD.Geometry;

/// <summary>
/// A sparse voxel volume: integer voxel coordinates with <see cref="Channels"/> float attributes each, on a regular grid whose
/// voxel <c>i</c> spans <c>[i, i + 1)</c> voxel units from <see cref="Origin"/> (voxel centers sit at <c>i + 0.5</c>).
/// Immutable after construction; lookups are O(1) through an open-addressing hash.
/// </summary>
public sealed class SparseAttributeVolume
{
    private const long EmptyKey = -1;
    private readonly long[] _keys;
    private readonly int[] _slots;
    private readonly int _mask;

    /// <summary>Creates a volume from voxel coordinates and per-voxel attributes (N voxels, <paramref name="channels"/> values each).</summary>
    /// <param name="coords">Voxel coordinates, x,y,z interleaved.</param>
    /// <param name="attributes">Voxel-major attribute values.</param>
    /// <param name="channels">Attribute values per voxel.</param>
    /// <param name="gridSize">Voxels per axis (x, y, z); every coordinate must lie inside it.</param>
    /// <param name="origin">World position of the grid corner (voxel 0 starts here), the upstream <c>aabb[0]</c>.</param>
    /// <param name="voxelSize">Edge length of one voxel in world units.</param>
    public SparseAttributeVolume(int[] coords, float[] attributes, int channels, (int X, int Y, int Z) gridSize, Vector3 origin,
        float voxelSize)
    {
        ArgumentNullException.ThrowIfNull(coords);
        ArgumentNullException.ThrowIfNull(attributes);
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels), "A volume needs at least one channel.");
        if (coords.Length % 3 != 0)
            throw new ArgumentException("Coordinates must be x,y,z triples.", nameof(coords));
        int count = coords.Length / 3;
        if ((long)count * channels != attributes.Length)
            throw new ArgumentException($"Expected {(long)count * channels} attribute values, got {attributes.Length}.", nameof(attributes));
        if (gridSize.X <= 0 || gridSize.Y <= 0 || gridSize.Z <= 0)
            throw new ArgumentOutOfRangeException(nameof(gridSize), "Grid size must be positive on every axis.");
        if (!(voxelSize > 0f) || !float.IsFinite(voxelSize))
            throw new ArgumentOutOfRangeException(nameof(voxelSize), "Voxel size must be positive and finite.");

        Coords = coords;
        Attributes = attributes;
        Channels = channels;
        GridSize = gridSize;
        Origin = origin;
        VoxelSize = voxelSize;

        int capacity = 16;
        while (capacity < count * 2L)
            capacity <<= 1;
        _mask = capacity - 1;
        _keys = new long[capacity];
        _slots = new int[capacity];
        Array.Fill(_keys, EmptyKey);
        for (int i = 0; i < count; i++)
        {
            int x = coords[3 * i], y = coords[3 * i + 1], z = coords[3 * i + 2];
            if ((uint)x >= (uint)gridSize.X || (uint)y >= (uint)gridSize.Y || (uint)z >= (uint)gridSize.Z)
                throw new ArgumentException($"Voxel {i} ({x},{y},{z}) lies outside the grid.", nameof(coords));
            if (!Insert(Key(x, y, z), i))
                throw new ArgumentException($"Voxel ({x},{y},{z}) appears more than once.", nameof(coords));
        }
    }

    /// <summary>Voxel coordinates, x,y,z interleaved.</summary>
    public int[] Coords { get; }

    /// <summary>Per-voxel attributes, voxel-major (<c>voxel * Channels + channel</c>).</summary>
    public float[] Attributes { get; }

    /// <summary>Attribute values per voxel.</summary>
    public int Channels { get; }

    /// <summary>Grid extent in voxels.</summary>
    public (int X, int Y, int Z) GridSize { get; }

    /// <summary>World position of the grid's minimum corner.</summary>
    public Vector3 Origin { get; }

    /// <summary>World edge length of one voxel.</summary>
    public float VoxelSize { get; }

    /// <summary>Number of occupied voxels.</summary>
    public int Count => Coords.Length / 3;

    /// <summary>Finds the voxel at integer coordinate (x,y,z); false when it is empty or outside the grid.</summary>
    public bool TryFindIndex(int x, int y, int z, out int index)
    {
        index = -1;
        if ((uint)x >= (uint)GridSize.X || (uint)y >= (uint)GridSize.Y || (uint)z >= (uint)GridSize.Z)
            return false;
        long key = Key(x, y, z);
        int slot = Hash(key) & _mask;
        while (true)
        {
            long stored = _keys[slot];
            if (stored == EmptyKey)
                return false;
            if (stored == key)
            {
                index = _slots[slot];
                return true;
            }
            slot = (slot + 1) & _mask;
        }
    }

    private long Key(int x, int y, int z) => x + (long)GridSize.X * (y + (long)GridSize.Y * z);

    private static int Hash(long key)
    {
        ulong h = (ulong)key * 0x9E3779B97F4A7C15UL;
        return (int)(h >> 33);
    }

    private bool Insert(long key, int index)
    {
        int slot = Hash(key) & _mask;
        while (_keys[slot] != EmptyKey)
        {
            if (_keys[slot] == key)
                return false;
            slot = (slot + 1) & _mask;
        }
        _keys[slot] = key;
        _slots[slot] = index;
        return true;
    }
}
