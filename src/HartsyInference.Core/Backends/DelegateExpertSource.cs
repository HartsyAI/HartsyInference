namespace HartsyInference.Core.Backends;

/// <summary>Adapts an existing resolver (a GGUF stacked-expert split, a checkpoint shard bank) to <see cref="IExpertSource"/> without copying it.</summary>
public sealed class DelegateExpertSource : IExpertSource
{
    private readonly Func<ExpertKey, ExpertWeights> _resolve;

    /// <summary>Wraps <paramref name="resolve"/>, which runs at most once per expert when used through an <see cref="ExpertBank"/>.</summary>
    public DelegateExpertSource(ExpertBacking backing, Func<ExpertKey, ExpertWeights> resolve)
    {
        ArgumentNullException.ThrowIfNull(resolve);
        Backing = backing;
        _resolve = resolve;
    }

    /// <inheritdoc/>
    public ExpertBacking Backing { get; }

    /// <inheritdoc/>
    public ExpertWeights Resolve(ExpertKey key) => _resolve(key);
}
