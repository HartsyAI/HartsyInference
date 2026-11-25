using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.CrossFrame;

/// <summary>
/// The unbuffered form of <see cref="CrossFrameTokens"/> (upstream <c>KVCache</c>): per cross-frame block, a FIFO of the last <see cref="Window"/>
/// frames, the current one included. Each block keeps a ring of the frames' prefix tokens and a head-major key/value cache <c>[1, heads, prefix +
/// Window·N, D]</c> whose first <c>prefix</c> rows hold the keys and values of the pooled prefix (the mean of the visible frames' prefixes) and
/// whose slot <c>s</c> holds the patch keys and values of a frame. The current frame's patches attend to the first <see cref="LiveRows"/> rows.
/// Slots fill in arrival order and then wrap, so the oldest frame is overwritten; attention has no temporal position, so slot order changes only
/// rounding. Per frame and block, call <see cref="UpdatePrefix"/>, then <see cref="Update"/>. Every write and read is a backend op
/// (<see cref="IBackend.CopyStrided"/>, <see cref="IBackend.AccumulateStrided"/>), so on a GPU backend the caches stay on the device and a frame
/// moves no key or value rows through the host.
/// </summary>
/// <remarks>
/// A frame that fails part-way leaves some blocks one frame ahead of the others; the ring is then unusable and must be replaced
/// (<c>DepthPro2VideoStream</c> refuses further frames after such a failure). MatrixGame2's sliding window is not reused here: it re-encodes its
/// context frames each step under a block-causal mask with temporal RoPE positions and keeps no key/value rows, while this ring keeps
/// position-free frame slots behind a pooled-prefix block that is rewritten every frame.
/// </remarks>
public sealed class CrossFrameKvRing : IDisposable
{
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
    /// pooled prefix, their mean <c>[1, P, C]</c> over the frames visible to this one (the caller owns it).
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
            entry = new Entry { Prefixes = Zeroed(backend, new TensorShape(Window, prefix.Shape[1], prefix.Shape[2])) };
            _blocks[block] = entry;
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
    /// D]</c>) ahead of the slots, then counts the frame as seen. Returns how many rows of <see cref="Keys"/> and <see cref="Values"/> the frame
    /// attends to.
    /// </summary>
    public int Update(IBackend backend, int block, Tensor key, Tensor value, Tensor pooledKey, Tensor pooledValue)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(backend);
        if (!_blocks.TryGetValue(block, out Entry? entry))
            throw new InvalidOperationException($"Call UpdatePrefix for block {block} before Update.");
        CheckRows(key, nameof(key));
        CheckRows(pooledKey, nameof(pooledKey));
        if (!value.Shape.Equals(key.Shape) || value.DType != DType.F32)
            throw new ArgumentException($"The values must match the keys {key.Shape}; got {value.DType} {value.Shape}.", nameof(value));
        if (!pooledValue.Shape.Equals(pooledKey.Shape) || pooledValue.DType != DType.F32)
            throw new ArgumentException(
                $"The pooled values must match the pooled keys {pooledKey.Shape}; got {pooledValue.DType} {pooledValue.Shape}.", nameof(pooledValue));
        int patches = (int)key.Shape[1], heads = (int)key.Shape[2], dim = (int)key.Shape[3], prefix = (int)pooledKey.Shape[1];
        if (pooledKey.Shape[2] != heads || pooledKey.Shape[3] != dim)
            throw new ArgumentException($"The pooled keys must have {heads} heads of {dim}; got {pooledKey.Shape}.", nameof(pooledKey));
        if (entry.Keys is null)
        {
            TensorShape shape = new(1, heads, prefix + (long)Window * patches, dim);
            entry.Keys = Zeroed(backend, shape);
            entry.Values = Zeroed(backend, shape);
            entry.Patches = patches;
            entry.Prefix = prefix;
        }
        else if (entry.Patches != patches || entry.Prefix != prefix || entry.Keys.Shape[1] != heads || entry.Keys.Shape[3] != dim)
        {
            throw new ArgumentException(
                $"Block {block} caches {entry.Prefix} + {Window} x {entry.Patches} rows of {entry.Keys.Shape[1]} heads; got {key.Shape}.",
                nameof(key));
        }

        int slot = entry.Seen % Window;
        WriteRows(backend, entry.Keys, key, prefix + slot * patches);
        WriteRows(backend, entry.Values!, value, prefix + slot * patches);
        WriteRows(backend, entry.Keys, pooledKey, 0);
        WriteRows(backend, entry.Values!, pooledValue, 0);
        entry.Seen++;
        return prefix + Math.Min(entry.Seen, Window) * patches;
    }

    /// <summary>Block <paramref name="block"/>'s head-major key cache; borrowed, valid until the ring is disposed.</summary>
    public Tensor Keys(int block) => Get(block).Keys ?? throw new InvalidOperationException($"Block {block} has no keys yet.");

    /// <summary>Block <paramref name="block"/>'s head-major value cache; borrowed, valid until the ring is disposed.</summary>
    public Tensor Values(int block) => Get(block).Values ?? throw new InvalidOperationException($"Block {block} has no values yet.");

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
            entry.Prefixes.Dispose();
            entry.Keys?.Dispose();
            entry.Values?.Dispose();
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

    // Token-major [1, S, heads, D] rows into head-major cache rows [first, first + S) of every head, as one strided copy on the backend.
    private static void WriteRows(IBackend backend, Tensor cache, Tensor rows, int first)
    {
        long count = rows.Shape[1], heads = rows.Shape[2], dim = rows.Shape[3], capacity = cache.Shape[2];
        StridedView destination = new([count, heads, dim], [dim, capacity * dim, 1], first * dim);
        StridedView source = new([count, heads, dim], [heads * dim, dim, 1], 0);
        backend.CopyStrided(cache, rows, new StridedCopySpec(destination, source));
    }

    /// <summary>The first <paramref name="rows"/> rows of every head of a head-major cache <c>[1, heads, capacity, D]</c>
    /// (<see cref="Keys"/> or <see cref="Values"/>), as a new <c>[1, heads, rows, D]</c> tensor (one strided copy on the backend); the caller owns
    /// it.</summary>
    public static Tensor LiveCopy(IBackend backend, Tensor cache, int rows)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(cache);
        if (cache.DType != DType.F32 || cache.Shape.Rank != 4 || cache.Shape[0] != 1)
            throw new ArgumentException($"The cache must be F32 head-major [1, heads, capacity, D]; got {cache.DType} {cache.Shape}.", nameof(cache));
        long heads = cache.Shape[1], capacity = cache.Shape[2], dim = cache.Shape[3];
        if (rows < 1 || rows > capacity)
            throw new ArgumentOutOfRangeException(nameof(rows), rows, $"A cache of {capacity} rows has no {rows} live rows.");
        Tensor live = new(new TensorShape(1, heads, rows, dim), DType.F32);
        backend.CopyStrided(live, cache, StridedCopySpec.Gather(new StridedView([heads, rows, dim], [capacity * dim, dim, 1], 0)));
        return live;
    }

    // A cache allocated and zeroed on the device by one broadcast copy, so the first in-place scatter into it finds a device buffer instead of
    // uploading a host buffer the size of the whole cache. Rows no frame has written are never read.
    private static Tensor Zeroed(IBackend backend, TensorShape shape)
    {
        Tensor cache = new(shape, DType.F32);
        try
        {
            using Tensor zero = new(new TensorShape(1), DType.F32);
            zero.AsSpan<float>().Clear();
            StridedCopies.FillFrom(backend, cache, zero);
            return cache;
        }
        catch
        {
            cache.Dispose();
            throw;
        }
    }

    private sealed class Entry
    {
        public required Tensor Prefixes { get; init; }
        public Tensor? Keys { get; set; }
        public Tensor? Values { get; set; }
        public int Patches { get; set; }
        public int Prefix { get; set; }
        public int Seen { get; set; }
    }
}
