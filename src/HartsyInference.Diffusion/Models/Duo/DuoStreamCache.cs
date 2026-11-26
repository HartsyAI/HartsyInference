using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>The committed history of a Duo denoise: one <see cref="KvSegmentStore"/> layer per block, with one segment per committed group.
/// Keys are post-QK-norm and post-RoPE; values are as the V projection produces them. A commit keeps only the reference, shared and clean
/// groups, as upstream <c>retain</c> does. Segments are device tensors, head-major <c>[1, heads, rows, head dim]</c>, token axis 2. Eviction
/// drops a whole group's segment from every layer, so reference and shared rows stay.</summary>
public sealed class DuoStreamCache : IDisposable
{
    private const int TokenAxis = 2;

    private readonly IBackend _backend;
    private readonly KvSegmentStore _store;
    private readonly List<DuoGroup> _groups = [];
    private int _length;
    private int _disposed;

    /// <param name="backend">Backend that owns the segments.</param>
    /// <param name="blocks">Transformer blocks, one store layer each.</param>
    /// <param name="width">Row width of each block's history: heads times head dimension.</param>
    /// <param name="capacityRows">Rows the history can hold. A commit beyond it throws.</param>
    public DuoStreamCache(IBackend backend, int blocks, int width, int capacityRows)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentOutOfRangeException.ThrowIfNegative(blocks);
        ArgumentOutOfRangeException.ThrowIfNegative(width);
        ArgumentOutOfRangeException.ThrowIfNegative(capacityRows);
        _backend = backend;
        _store = new KvSegmentStore(backend, Math.Max(blocks, 1), TokenAxis);
        Blocks = blocks;
        Width = width;
        CapacityRows = capacityRows;
    }

    /// <summary>Transformer blocks this cache serves.</summary>
    public int Blocks { get; }

    /// <summary>Row width of each block's history.</summary>
    public int Width { get; }

    /// <summary>Committed rows. Every block holds the same count.</summary>
    public int Length => _length;

    /// <summary>Rows the history can hold.</summary>
    public int CapacityRows { get; }

    /// <summary>The groups that own the committed rows, in cache coordinates.</summary>
    public IReadOnlyList<DuoGroup> Groups => _groups;

    /// <summary>The segments of every block; a forward reads them with <see cref="KvSegmentStore.Join"/>.</summary>
    internal KvSegmentStore Store => _store;

    /// <summary>Appends the groups of one forward. <paramref name="keys"/> and <paramref name="values"/> hold one head-major tensor per block,
    /// <c>[1, heads, rows, head dim]</c>, indexed by the forward's own rows; <paramref name="layoutGroups"/> name the rows to keep in those
    /// coordinates. Every slice is made before the store is touched, so a refused commit leaves the cache unchanged.</summary>
    internal void Commit(IReadOnlyList<Tensor> keys, IReadOnlyList<Tensor> values, IReadOnlyList<DuoGroup> layoutGroups)
    {
        ArgumentNullException.ThrowIfNull(keys);
        ArgumentNullException.ThrowIfNull(values);
        ArgumentNullException.ThrowIfNull(layoutGroups);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (keys.Count != Blocks || values.Count != Blocks)
        {
            throw new ArgumentException($"A commit needs one key and one value tensor per block ({Blocks}); got {keys.Count} and {values.Count}.");
        }

        int rows = 0;
        foreach (DuoGroup g in layoutGroups)
        {
            if (g.Start < 0 || g.End <= g.Start)
            {
                throw new ArgumentException($"Group [{g.Start}, {g.End}) is empty or negative.", nameof(layoutGroups));
            }
            rows += g.End - g.Start;
        }
        if (rows > CapacityRows - Length)
        {
            throw new InvalidOperationException($"Committing {rows} rows to {Length} would exceed the {CapacityRows}-row capacity.");
        }
        for (int b = 0; b < Blocks; b++)
        {
            if (keys[b].Shape.Rank != 4 || values[b].Shape.Rank != 4)
            {
                throw new ArgumentException($"Block {b} needs head-major [1, heads, rows, head dim] tensors.");
            }
            foreach (DuoGroup g in layoutGroups)
            {
                if (g.End > keys[b].Shape[TokenAxis] || g.End > values[b].Shape[TokenAxis])
                {
                    throw new ArgumentException($"Group [{g.Start}, {g.End}) lies outside block {b}'s {keys[b].Shape[TokenAxis]} rows.", nameof(layoutGroups));
                }
            }
        }

        // Slice everything first; a failure here disposes the slices and leaves the cache as it was.
        List<(Tensor Keys, Tensor Values)> slices = [];
        try
        {
            foreach (DuoGroup g in layoutGroups)
            {
                for (int b = 0; b < Blocks; b++)
                {
                    slices.Add((SliceRows(keys[b], g.Start, g.End - g.Start), SliceRows(values[b], g.Start, g.End - g.Start)));
                }
            }
        }
        catch
        {
            foreach ((Tensor k, Tensor v) in slices)
            {
                k.Dispose();
                v.Dispose();
            }
            throw;
        }

        int cursor = Length;
        int[] prior = new int[Blocks];
        for (int b = 0; b < Blocks; b++)
        {
            prior[b] = _store.Count(b);
        }
        int index = 0;
        try
        {
            foreach (DuoGroup g in layoutGroups)
            {
                for (int b = 0; b < Blocks; b++, index++)
                {
                    _store.Append(b, slices[index].Keys, slices[index].Values);
                }
            }
        }
        catch
        {
            // The store takes ownership of a pair even when its write throws, so the slice at `index` is already its own. Drop the
            // segments this commit added, and free the slices after the failed one, which the store never saw.
            for (int b = 0; b < Blocks; b++)
            {
                while (_store.Count(b) > prior[b])
                {
                    _store.RemoveRange(b, _store.Count(b) - 1, 1);
                }
            }
            for (int i = index + 1; i < slices.Count; i++)
            {
                slices[i].Keys.Dispose();
                slices[i].Values.Dispose();
            }
            throw;
        }

        foreach (DuoGroup g in layoutGroups)
        {
            _groups.Add(g with { Start = cursor, End = cursor + (g.End - g.Start) });
            cursor += g.End - g.Start;
        }
        _length = cursor;
    }

    /// <summary>Drops the oldest committed clean block: its segment in every layer, and its group. Reference and shared groups are never
    /// dropped. Returns the rows dropped, or 0 when no clean block is held.</summary>
    public int EvictOldestClean()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        int index = _groups.FindIndex(g => g.Kind == DuoGroupKind.Clean);
        if (index < 0)
        {
            return 0;
        }
        DuoGroup group = _groups[index];
        int rows = group.End - group.Start;
        for (int b = 0; b < Blocks; b++)
        {
            _store.RemoveRange(b, index, 1);
        }
        _groups.RemoveAt(index);
        for (int g = 0; g < _groups.Count; g++)
        {
            DuoGroup later = _groups[g];
            if (later.Start >= group.End)
            {
                _groups[g] = later with { Start = later.Start - rows, End = later.End - rows };
            }
        }
        _length -= rows;
        return rows;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        _store.Dispose();
    }

    /// <summary>Copies rows <c>[start, start + length)</c> of a head-major tensor into a new tensor the caller owns.</summary>
    private Tensor SliceRows(Tensor input, int start, int length)
    {
        Span<long> dims = stackalloc long[input.Shape.Rank];
        input.Shape.CopyDimsTo(dims);
        dims[TokenAxis] = length;
        Tensor output = new(new TensorShape(dims), input.DType);
        try
        {
            StridedCopies.SliceAxis(_backend, output, input, TokenAxis, start, length);
        }
        catch
        {
            output.Dispose();
            throw;
        }
        return output;
    }
}
