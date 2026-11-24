namespace HartsyInference.LLM.Maxwell;

/// <summary>Per-sequence caches of every Maxwell decoder layer: the keys and values after q/k norm and RoPE, and the indexer's keys, all from token 0.
/// Host F32 arrays that grow with the sequence.</summary>
public sealed class MaxwellSequenceState
{
    private readonly float[][] _keys;
    private readonly float[][] _values;
    private readonly float[][] _indexKeys;
    private readonly int _kvStride;
    private readonly int _indexStride;

    /// <param name="config">The model's config.</param>
    /// <param name="capacity">Longest sequence this state will hold.</param>
    public MaxwellSequenceState(MaxwellConfig config, int capacity)
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

    /// <summary>Forgets every token.</summary>
    public void Reset()
    {
        Length = 0;
        Faulted = false;
    }

    /// <summary>Rolls the sequence back to <paramref name="length"/> tokens; later entries are overwritten by the next forward.</summary>
    public void Truncate(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, Length);
        Length = length;
    }

    internal void MarkFaulted() => Faulted = true;

    /// <summary>Makes room for <paramref name="tokens"/> entries in every cache of <paramref name="layer"/> and returns them.</summary>
    internal (float[] Keys, float[] Values, float[] IndexKeys) Reserve(int layer, int tokens)
    {
        _keys[layer] = Grow(_keys[layer], tokens, _kvStride);
        _values[layer] = Grow(_values[layer], tokens, _kvStride);
        _indexKeys[layer] = Grow(_indexKeys[layer], tokens, _indexStride);
        return (_keys[layer], _values[layer], _indexKeys[layer]);
    }

    private static float[] Grow(float[] cache, int tokens, int stride)
    {
        long needed = (long)tokens * stride;
        if (cache.Length >= needed) return cache;
        long next = Math.Max(needed, Math.Min(cache.Length * 2L, (long)int.MaxValue / 2));
        Array.Resize(ref cache, checked((int)next));
        return cache;
    }
}
