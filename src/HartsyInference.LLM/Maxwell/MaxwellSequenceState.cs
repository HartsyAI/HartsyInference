using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Maxwell;

/// <summary>Per-sequence caches of every Maxwell decoder layer: the keys and values after q/k norm and RoPE, and the indexer's keys, all from token 0.
/// Host F32 arrays that grow with the sequence. Optionally also a device key/value cache for the dense-range attention
/// (<see cref="MaxwellLoadOptions.DeviceDenseAttention"/>), which holds the keys and values of the first <see cref="DenseLength"/> tokens.</summary>
public sealed class MaxwellSequenceState : IDisposable
{
    private readonly float[][] _keys;
    private readonly float[][] _values;
    private readonly float[][] _indexKeys;
    private readonly int _kvStride;
    private readonly int _indexStride;
    private int _disposed;

    /// <param name="config">The model's config.</param>
    /// <param name="capacity">Longest sequence this state will hold.</param>
    internal MaxwellSequenceState(MaxwellConfig config, int capacity)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        Capacity = capacity;
        _kvStride = config.NumKeyValueHeads * config.HeadDim;
        _indexStride = config.Sparse.IndexerHeadDim;
        _keys = new float[config.NumHiddenLayers][];
        _values = new float[config.NumHiddenLayers][];
        _indexKeys = new float[config.NumHiddenLayers][];
        Array.Fill(_keys, Array.Empty<float>());
        Array.Fill(_values, Array.Empty<float>());
        Array.Fill(_indexKeys, Array.Empty<float>());
    }

    /// <summary>Tokens committed to the sequence.</summary>
    public int Length { get; internal set; }

    /// <summary>Longest sequence the state will hold.</summary>
    public int Capacity { get; }

    /// <summary>Whether a forward failed partway through, leaving the layer caches ahead of <see cref="Length"/>.</summary>
    public bool Faulted { get; private set; }

    /// <summary>The device key/value cache of the dense-range attention, or null when the state keeps host caches only.</summary>
    internal KvCache? Dense { get; private set; }

    /// <summary>Tokens held in <see cref="Dense"/>, for every layer. Dense attention runs only when this equals the forward's start position.</summary>
    internal int DenseLength { get; private set; }

    /// <summary>Forgets every token.</summary>
    public void Reset()
    {
        Length = 0;
        Faulted = false;
        ResetDense();
    }

    /// <summary>Rolls the sequence back to <paramref name="length"/> tokens; later entries are overwritten by the next forward.</summary>
    public void Truncate(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, Length);
        if (length != Length) ResetDense();
        Length = length;
    }

    internal void MarkFaulted() => Faulted = true;

    /// <summary>Gives the state its device key/value cache for the dense-range attention; called once, before the first forward.</summary>
    internal void AttachDense(KvCache cache)
    {
        ArgumentNullException.ThrowIfNull(cache);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        Dense?.Dispose();
        Dense = cache;
        DenseLength = 0;
    }

    /// <summary>Whether a forward of <paramref name="tokens"/> tokens from <paramref name="startPos"/> runs its attention dense: the state has the device
    /// cache, every query is below <paramref name="denseLimit"/>, and the cache holds exactly the tokens before <paramref name="startPos"/>.</summary>
    internal bool UsesDense(int startPos, int tokens, int denseLimit) =>
        Dense is not null && startPos + tokens <= denseLimit && DenseLength == startPos;

    /// <summary>Records that a forward has appended its tokens to the device cache, once every layer has run. A forward that did not run dense
    /// leaves <see cref="DenseLength"/> behind, so later dense forwards are not taken until a reset.</summary>
    internal void CommitDense(int startPos, int tokens, int denseLimit)
    {
        if (UsesDense(startPos, tokens, denseLimit)) DenseLength = startPos + tokens;
    }

    /// <summary>Makes room for <paramref name="tokens"/> entries in every cache of <paramref name="layer"/> and returns them.</summary>
    internal (float[] Keys, float[] Values, float[] IndexKeys) Reserve(int layer, int tokens)
    {
        _keys[layer] = Grow(_keys[layer], tokens, _kvStride, Capacity);
        _values[layer] = Grow(_values[layer], tokens, _kvStride, Capacity);
        _indexKeys[layer] = Grow(_indexKeys[layer], tokens, _indexStride, Capacity);
        return (_keys[layer], _values[layer], _indexKeys[layer]);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Dense?.Dispose();
        Dense = null;
    }

    // The device cache is cleared with the sequence; the next dense forward needs it to hold the tokens before its start.
    private void ResetDense()
    {
        Dense?.Reset();
        DenseLength = 0;
    }

    // Doubles, but never past the capacity the state was made for: the caches are sized for the longest sequence at most once.
    private static float[] Grow(float[] cache, int tokens, int stride, int capacity)
    {
        long needed = (long)tokens * stride;
        if (cache.Length >= needed) return cache;
        long next = Math.Max(needed, Math.Min(cache.Length * 2L, (long)capacity * stride));
        Array.Resize(ref cache, checked((int)next));
        return cache;
    }
}
