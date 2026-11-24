using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.CrossFrame;

/// <summary>
/// The unbuffered form of <see cref="CrossFrameTokens"/> (upstream <c>KVCache</c>): per cross-frame block, a FIFO of the last <see cref="Window"/>
/// frames, the current one included. Each block keeps a ring of the frames' prefix tokens and a head-major key/value cache <c>[1, heads, prefix +
/// Window·N, D]</c> whose first <c>prefix</c> rows hold the keys and values of the pooled prefix (the mean of the visible frames' prefixes) and
/// whose slot <c>s</c> holds the patch keys and values of a frame. The current frame's patches attend to the first <see cref="LiveRows"/> rows.
/// Slots fill in arrival order and then wrap, so the oldest frame is overwritten; attention has no temporal position, so slot order changes only
/// rounding. Per frame and block, call <see cref="UpdatePrefix"/>, then <see cref="Update"/>.
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
    public Tensor UpdatePrefix(int block, Tensor prefix)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(prefix);
        if (prefix.DType != DType.F32 || prefix.Shape.Rank != 3 || prefix.Shape[0] != 1)
            throw new ArgumentException($"The prefix must be F32 [1, P, C]; got {prefix.DType} {prefix.Shape}.", nameof(prefix));
        if (!_blocks.TryGetValue(block, out Entry? entry))
        {
            entry = new Entry { Prefixes = new Tensor(new TensorShape(Window, prefix.Shape[1], prefix.Shape[2]), DType.F32) };
            _blocks[block] = entry;
        }
        else if (entry.Prefixes.Shape[1] != prefix.Shape[1] || entry.Prefixes.Shape[2] != prefix.Shape[2])
        {
            throw new ArgumentException(
                $"Block {block} caches prefixes of [{entry.Prefixes.Shape[1]}, {entry.Prefixes.Shape[2]}]; got {prefix.Shape}.", nameof(prefix));
        }

        int width = (int)prefix.ElementCount;
        Span<float> ring = entry.Prefixes.AsSpan<float>();
        prefix.AsReadOnlySpan<float>().CopyTo(ring.Slice(entry.Seen % Window * width, width));
        int visible = Math.Min(entry.Seen + 1, Window);
        Tensor pooled = new(prefix.Shape, DType.F32);
        Span<float> mean = pooled.AsSpan<float>();
        mean.Clear();
        for (int s = 0; s < visible; s++)
        {
            ReadOnlySpan<float> slot = ring.Slice(s * width, width);
            for (int i = 0; i < width; i++) mean[i] += slot[i];
        }

        for (int i = 0; i < width; i++) mean[i] /= visible;
        return pooled;
    }

    /// <summary>
    /// Stores the current frame's patch keys and values (token-major <c>[1, N, heads, D]</c>) in its slot and the pooled prefix's (<c>[1, P, heads,
    /// D]</c>) ahead of the slots, then counts the frame as seen. Returns how many rows of <see cref="Keys"/> and <see cref="Values"/> the frame
    /// attends to.
    /// </summary>
    public int Update(int block, Tensor key, Tensor value, Tensor pooledKey, Tensor pooledValue)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
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
            entry.Keys = new Tensor(shape, DType.F32);
            entry.Values = new Tensor(shape, DType.F32);
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
        WriteRows(entry.Keys, key, prefix + slot * patches);
        WriteRows(entry.Values!, value, prefix + slot * patches);
        WriteRows(entry.Keys, pooledKey, 0);
        WriteRows(entry.Values!, pooledValue, 0);
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

    // Token-major [1, S, heads, D] rows into head-major cache rows [first, first + S) of every head.
    private static void WriteRows(Tensor cache, Tensor rows, int first)
    {
        int count = (int)rows.Shape[1], heads = (int)rows.Shape[2], dim = (int)rows.Shape[3], capacity = (int)cache.Shape[2];
        ReadOnlySpan<float> src = rows.AsReadOnlySpan<float>();
        Span<float> dst = cache.AsSpan<float>();
        for (int s = 0; s < count; s++)
            for (int h = 0; h < heads; h++)
                src.Slice((s * heads + h) * dim, dim).CopyTo(dst.Slice((h * capacity + first + s) * dim, dim));
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
