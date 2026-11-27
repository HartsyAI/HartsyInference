using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.CrossFrame;

/// <summary>
/// The unbuffered form of <see cref="CrossFrameTokens"/> (upstream <c>KVCache</c>): per cross-frame block, a FIFO of the last <see cref="Window"/>
/// frames, the current one included. Each block keeps a ring of the frames' prefix tokens and, in a <see cref="KvSegmentStore"/>, the keys and
/// values as head-major <c>[1, heads, rows, D]</c> segments: segment 0 holds those of the pooled prefix (the mean of the visible frames' prefixes)
/// and segment <c>1 + s</c> those of the frame in slot <c>s</c>. <see cref="Read"/> copies the live ones along the rows into what the current frame's
/// patches attend to. Slots fill in arrival order and then wrap, so the oldest frame is replaced; attention has no temporal position, so slot order
/// changes only rounding. Per frame and block, call <see cref="UpdatePrefix"/>, then <see cref="Update"/>, then <see cref="Read"/>. Every write and
/// read is a backend op and every kept tensor is pinned, so on a GPU backend the caches stay on the device.
/// </summary>
/// <remarks>
/// A frame that fails part-way leaves some blocks one frame ahead of the others, and an <see cref="Update"/> that fails while it stores can leave
/// its block's pooled rows ahead of its slots; the ring is then unusable and must be replaced (<c>DepthPro2VideoStream</c> refuses further frames
/// after such a failure). MatrixGame2's sliding window is not reused here: it re-encodes its
/// context frames each step under a block-causal mask with temporal RoPE positions and keeps no key/value rows, while this ring keeps
/// position-free frame slots behind a pooled-prefix block that is rewritten every frame.
/// </remarks>
public sealed class CrossFrameKvRing : IDisposable
{
    private const int TokenAxis = 2;
    private const int PooledSlot = 0;

    private readonly Dictionary<int, Entry> _blocks = [];
    private bool _disposed;

    /// <param name="window">The frames attended together, the current one included; at least 1.</param>
    public CrossFrameKvRing(int window)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(window, 1);
        Window = window;
    }

    /// <summary>The frames each block keeps, the current one included.</summary>
    public int Window { get; }

    /// <summary>How many frames block <paramref name="block"/> has seen.</summary>
    public int Seen(int block) => _blocks.TryGetValue(block, out Entry? entry) ? entry.Seen : 0;

    /// <summary>
    /// Stores the current frame's prefix tokens <paramref name="prefix"/> <c>[1, P, C]</c> in block <paramref name="block"/>'s ring and returns the
    /// pooled prefix, their mean <c>[1, P, C]</c> over the frames visible to this one (the caller owns it). The first call for a block fixes the
    /// backend that holds its caches.
    /// </summary>
    public Tensor UpdatePrefix(IBackend backend, int block, Tensor prefix)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(prefix);
        if (prefix.DType != DType.F32 || prefix.Shape.Rank != 3 || prefix.Shape[0] != 1)
            throw new ArgumentException($"The prefix must be F32 [1, P, C]; got {prefix.DType} {prefix.Shape}.", nameof(prefix));
        if (!_blocks.TryGetValue(block, out Entry? entry))
        {
            entry = NewEntry(backend, new TensorShape(Window, prefix.Shape[1], prefix.Shape[2]));
            _blocks[block] = entry;
        }
        else if (!ReferenceEquals(entry.Backend, backend))
        {
            throw new ArgumentException($"Block {block}'s caches live on another backend.", nameof(backend));
        }
        else if (entry.Prefixes.Shape[1] != prefix.Shape[1] || entry.Prefixes.Shape[2] != prefix.Shape[2])
        {
            throw new ArgumentException(
                $"Block {block} caches prefixes of [{entry.Prefixes.Shape[1]}, {entry.Prefixes.Shape[2]}]; got {prefix.Shape}.", nameof(prefix));
        }

        StridedCopies.ScatterAxis(backend, entry.Prefixes, prefix, 0, entry.Seen % Window);
        int visible = Math.Min(entry.Seen + 1, Window);

        // The mean of the visible slots, summed in slot order: the first slot, then each later one added in place.
        Tensor sum = new(prefix.Shape, DType.F32);
        try
        {
            StridedCopies.SliceAxis(backend, sum, entry.Prefixes, 0, 0, 1);
            StridedView whole = StridedView.Of(sum.Shape);
            for (int s = 1; s < visible; s++)
                backend.AccumulateStrided(sum, entry.Prefixes, new StridedCopySpec(whole, StridedView.Of(entry.Prefixes.Shape).Slice(0, s, 1)));
            if (visible == 1) return sum;
            Tensor pooled = new(prefix.Shape, DType.F32);
            backend.Scale(pooled, sum, 1f / visible);
            sum.Dispose();
            return pooled;
        }
        catch
        {
            sum.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Stores the current frame's patch keys and values (token-major <c>[1, N, heads, D]</c>) in its slot and the pooled prefix's (<c>[1, P, heads,
    /// D]</c>) ahead of the slots, then counts the frame as seen. The ring takes ownership of all four tensors, also when it throws: it keeps
    /// head-major copies, disposes the four, and frees the segments the copies replace. Returns how many rows <see cref="Read"/> now gives.
    /// </summary>
    public int Update(int block, Tensor key, Tensor value, Tensor pooledKey, Tensor pooledValue)
    {
        using (key)
        using (value)
        using (pooledKey)
        using (pooledValue)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!_blocks.TryGetValue(block, out Entry? entry))
                throw new InvalidOperationException($"Call UpdatePrefix for block {block} before Update.");
            CheckRows(key, nameof(key));
            CheckRows(pooledKey, nameof(pooledKey));
            ArgumentNullException.ThrowIfNull(value);
            ArgumentNullException.ThrowIfNull(pooledValue);
            if (!value.Shape.Equals(key.Shape) || value.DType != DType.F32)
                throw new ArgumentException($"The values must match the keys {key.Shape}; got {value.DType} {value.Shape}.", nameof(value));
            if (!pooledValue.Shape.Equals(pooledKey.Shape) || pooledValue.DType != DType.F32)
                throw new ArgumentException(
                    $"The pooled values must match the pooled keys {pooledKey.Shape}; got {pooledValue.DType} {pooledValue.Shape}.",
                    nameof(pooledValue));
            int patches = (int)key.Shape[1], heads = (int)key.Shape[2], dim = (int)key.Shape[3], prefix = (int)pooledKey.Shape[1];
            if (pooledKey.Shape[2] != heads || pooledKey.Shape[3] != dim)
                throw new ArgumentException($"The pooled keys must have {heads} heads of {dim}; got {pooledKey.Shape}.", nameof(pooledKey));
            if (prefix != entry.Prefixes.Shape[1])
                throw new ArgumentException($"Block {block} pools {entry.Prefixes.Shape[1]} prefix rows; got {pooledKey.Shape}.",
                    nameof(pooledKey));
            if (entry.Seen > 0 && (entry.Patches != patches || entry.Heads != heads || entry.Dim != dim))
                throw new ArgumentException(
                    $"Block {block} caches {entry.Prefix} + {Window} x {entry.Patches} rows of {entry.Heads} heads; got {key.Shape}.", nameof(key));

            // The pooled rows first: the store then never has an empty slot before a frame's.
            Store(entry, PooledSlot, pooledKey, pooledValue);
            Store(entry, 1 + entry.Seen % Window, key, value);
            (entry.Patches, entry.Prefix, entry.Heads, entry.Dim) = (patches, prefix, heads, dim);
            entry.Seen++;
            return LiveRows(block);
        }
    }

    /// <summary>The keys and values the frame last stored in block <paramref name="block"/> attends to, head-major <c>[1, heads, rows, D]</c>: the
    /// pooled prefix's rows, then the filled slots in slot order. Both are new tensors the caller owns.</summary>
    /// <exception cref="InvalidOperationException">The block has seen no frame.</exception>
    public (Tensor Keys, Tensor Values) Read(int block)
    {
        Entry entry = Get(block);
        if (entry.Seen == 0) throw new InvalidOperationException($"Block {block} has no keys and values yet.");
        // The pooled prefix is slot 0 and the filled slots follow it, so the store's join is the order the frame attends to.
        return entry.Store.Join(0, 0, 1 + Math.Min(entry.Seen, Window));
    }

    /// <summary>The rows the frame last stored in block <paramref name="block"/> attends to.</summary>
    public int LiveRows(int block)
    {
        Entry entry = Get(block);
        return entry.Prefix + Math.Min(entry.Seen, Window) * entry.Patches;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (Entry entry in _blocks.Values)
        {
            entry.Store.Dispose();
            entry.Backend.UnpinActivation(entry.Prefixes);
            entry.Prefixes.Dispose();
        }

        _blocks.Clear();
    }

    private Entry Get(int block)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _blocks.TryGetValue(block, out Entry? entry) ? entry : throw new InvalidOperationException($"Block {block} has seen no frame.");
    }

    private static void CheckRows(Tensor rows, string name)
    {
        ArgumentNullException.ThrowIfNull(rows, name);
        if (rows.DType != DType.F32 || rows.Shape.Rank != 4 || rows.Shape[0] != 1)
            throw new ArgumentException($"{name} must be F32 token-major [1, S, heads, D]; got {rows.DType} {rows.Shape}.", name);
    }

    // Head-major copies of token-major keys and values into a slot.
    private static void Store(Entry entry, int slot, Tensor keys, Tensor values)
    {
        Tensor headKeys = HeadMajor(entry.Backend, keys);
        Tensor headValues;
        try
        {
            headValues = HeadMajor(entry.Backend, values);
        }
        catch
        {
            headKeys.Dispose();
            throw;
        }

        entry.Store.Set(0, slot, headKeys, headValues);   // the store takes both, also when it throws
    }

    // [1, S, heads, D] to [1, heads, S, D].
    private static Tensor HeadMajor(IBackend backend, Tensor rows)
    {
        int count = (int)rows.Shape[1], heads = (int)rows.Shape[2], dim = (int)rows.Shape[3];
        Tensor result = new(new TensorShape(1, heads, count, dim), DType.F32);
        try
        {
            backend.Permute0213(result, rows, count, heads, dim);
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }

    // The prefix ring is allocated and zeroed on the device by one broadcast copy, so the first in-place scatter into it finds a device buffer
    // instead of uploading a host buffer. Slots no frame has written are never read.
    private static Entry NewEntry(IBackend backend, TensorShape prefixShape)
    {
        Tensor prefixes = new(prefixShape, DType.F32);
        try
        {
            using Tensor zero = new(new TensorShape(1), DType.F32);
            zero.AsSpan<float>().Clear();
            StridedCopies.FillFrom(backend, prefixes, zero);
            backend.PinActivation(prefixes);
            return new Entry { Backend = backend, Prefixes = prefixes, Store = new KvSegmentStore(backend, 1, TokenAxis) };
        }
        catch
        {
            backend.UnpinActivation(prefixes);
            prefixes.Dispose();
            throw;
        }
    }

    private sealed class Entry
    {
        public required IBackend Backend { get; init; }
        public required Tensor Prefixes { get; init; }
        public required KvSegmentStore Store { get; init; }
        public int Patches { get; set; }
        public int Prefix { get; set; }
        public int Heads { get; set; }
        public int Dim { get; set; }
        public int Seen { get; set; }
    }
}
