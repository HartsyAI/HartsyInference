using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Multimodal;

/// <summary>The projected, normalized, rotated vision keys and values a gated cross-attention decoder has seen so far, per cross layer.
/// Vision arrives in segments (one image, a batch of video frames, a frame pushed into a live session): each forward that brings new
/// vision tokens appends them once, and every later forward reads all of them without projecting anything again.</summary>
/// <remarks>
/// <para>Keys and values are head-major <c>[1, heads, capacity, head_dim]</c> F32 buffers on the backend that appended them; the first
/// <see cref="Length"/> rows are valid. A forward appends its new rows at <see cref="Length"/> in every cross layer and the caller
/// commits them with <see cref="Commit"/> once the whole forward has run. A forward that stops part-way commits nothing, and the
/// rows it wrote past <see cref="Length"/> are overwritten by the next one, so the cache never holds a half-appended segment.</para>
/// <para>Capacity doubles when a segment does not fit (the old rows are copied on the device). The caller owns the cache and
/// disposes it when the session ends.</para>
/// </remarks>
public sealed class CrossKvCache : IDisposable
{
    private readonly Tensor?[] _k;
    private readonly Tensor?[] _v;
    private readonly int _initialCapacity;
    private int _disposed;

    /// <param name="layers">The decoder's layer count (cross layers are indexed by their decoder layer).</param>
    /// <param name="initialCapacity">Rows to allocate on a layer's first append; at least the first segment's length.</param>
    public CrossKvCache(int layers, int initialCapacity = 0)
    {
        _k = new Tensor?[layers];
        _v = new Tensor?[layers];
        _initialCapacity = Math.Max(0, initialCapacity);
    }

    /// <summary>Vision tokens committed so far: the rows every cross layer holds and every forward attends.</summary>
    public int Length { get; private set; }

    /// <summary>The key and value buffers of <paramref name="layer"/>, or null before it has appended anything. Valid rows are the first
    /// <see cref="Length"/> plus whatever the running forward appended.</summary>
    public (Tensor K, Tensor V)? Get(int layer) => _k[layer] is { } k ? (k, _v[layer]!) : null;

    /// <summary>Writes <paramref name="k"/> and <paramref name="v"/> (<c>[1, heads, n, head_dim]</c>, borrowed) at row <see cref="Length"/>
    /// of <paramref name="layer"/>, growing its buffers when they are full.</summary>
    public void Append(IBackend backend, int layer, Tensor k, Tensor v)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        int heads = (int)k.Shape[1], n = (int)k.Shape[2], headDim = (int)k.Shape[3];
        int needed = Length + n;
        int capacity = _k[layer] is { } existing ? (int)existing.Shape[2] : 0;
        if (needed > capacity)
            Grow(backend, layer, heads, headDim, Math.Max(needed, Math.Max(_initialCapacity, capacity * 2)));
        backend.KvCacheAppend(_k[layer]!, k, Length);
        backend.KvCacheAppend(_v[layer]!, v, Length);
    }

    /// <summary>Makes the <paramref name="count"/> rows the last forward appended part of the cache.</summary>
    public void Commit(int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        Length += count;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        for (int i = 0; i < _k.Length; i++)
        {
            _k[i]?.Dispose();
            _v[i]?.Dispose();
            _k[i] = _v[i] = null;
        }
    }

    private void Grow(IBackend backend, int layer, int heads, int headDim, int capacity)
    {
        Tensor k = new(new TensorShape(1, heads, capacity, headDim), DType.F32);
        Tensor v = new(new TensorShape(1, heads, capacity, headDim), DType.F32);
        backend.ResidentAllocateKv(k);
        backend.ResidentAllocateKv(v);
        if (_k[layer] is { } oldK && _v[layer] is { } oldV)
        {
            // Copy the old rows (the whole old buffer: rows past Length are never read), then wait for the copy before the old
            // buffers are freed under it. Growth doubles the capacity, so it happens a handful of times per session.
            backend.KvCacheAppend(k, oldK, 0);
            backend.KvCacheAppend(v, oldV, 0);
            backend.Sync();
            oldK.Dispose();
            oldV.Dispose();
        }
        _k[layer] = k;
        _v[layer] = v;
    }
}
