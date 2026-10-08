namespace HartsyInference.Core.Backends;

/// <summary>One layer's routed experts, each <see cref="ExpertWeights"/> built once on first use so its tensor identity stays stable for the cache.</summary>
public sealed class ExpertBank
{
    private readonly Func<ExpertKey, ExpertWeights> _resolve;
    private readonly ExpertWeights?[] _created;
    private readonly object _gate = new();

    /// <summary>
    /// Creates a bank; <paramref name="resolve"/> runs at most once per expert and must return weights carrying the requested key.
    /// </summary>
    /// <param name="layer">Layer number within the bank.</param>
    /// <param name="count">Routed experts in the layer.</param>
    /// <param name="resolve">Produces one expert's weights.</param>
    /// <param name="bank">The bank this layer belongs to; 0 unless a model shares a cache with others.</param>
    public ExpertBank(int layer, int count, Func<ExpertKey, ExpertWeights> resolve, ushort bank)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(layer);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        ArgumentNullException.ThrowIfNull(resolve);
        Layer = layer;
        Bank = bank;
        Count = count;
        _resolve = resolve;
        _created = new ExpertWeights?[count];
    }

    /// <summary>The published three-argument form; bank 0. Kept so existing callers still compile and link.</summary>
    public ExpertBank(int layer, int count, Func<ExpertKey, ExpertWeights> resolve) : this(layer, count, resolve, 0)
    {
    }

    /// <summary>Creates a bank whose experts come from an <see cref="IExpertSource"/>.</summary>
    public static ExpertBank FromSource(IExpertSource source, int layer, int count, ushort bank = 0)
    {
        ArgumentNullException.ThrowIfNull(source);
        return new ExpertBank(layer, count, source.Resolve, bank);
    }

    /// <summary>Layer number.</summary>
    public int Layer { get; }

    /// <summary>The bank this layer belongs to.</summary>
    public ushort Bank { get; }

    /// <summary>The cache identity of this layer.</summary>
    public ExpertLayerKey LayerKey => new(Bank, Layer);

    /// <summary>Routed experts in the layer.</summary>
    public int Count { get; }

    /// <summary>How many experts have been resolved so far.</summary>
    public int ResolvedCount
    {
        get
        {
            lock (_gate) return _created.Count(static weights => weights is not null);
        }
    }

    /// <summary>The key of <paramref name="expert"/> in this layer.</summary>
    public ExpertKey Key(int expert)
    {
        if ((uint)expert >= (uint)Count)
            throw new ArgumentOutOfRangeException(nameof(expert), expert, $"Layer {Layer} has {Count} routed experts.");
        return new ExpertKey(Layer, expert, Bank);
    }

    /// <summary>The expert's weights; the same object every call.</summary>
    public ExpertWeights Get(int expert)
    {
        ExpertKey key = Key(expert);
        lock (_gate)
        {
            ExpertWeights? weights = _created[expert];
            if (weights is not null) return weights;
            weights = _resolve(key) ?? throw new InvalidOperationException($"Expert resolver returned null for {key}.");
            if (weights.Key != key)
                throw new InvalidOperationException($"Expert resolver returned weights for {weights.Key} when {key} was requested.");
            _created[expert] = weights;
            return weights;
        }
    }
}
