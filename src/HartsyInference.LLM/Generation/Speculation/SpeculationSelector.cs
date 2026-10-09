namespace HartsyInference.LLM.Generation.Speculation;

/// <summary>Picks which draft provider to use each round, by measured throughput, and disables providers
/// whose drafts the target keeps too rarely.</summary>
/// <remarks>
/// <para>Each round the caller calls <see cref="Select"/>, drafts with the returned provider, verifies, times the whole
/// round and reports it through <see cref="Record"/>. The selector has no clock and no randomness, so the same
/// sequence of measurements always yields the same decisions.</para>
/// <para>Selection: an enabled provider with no recorded round is chosen first, in registration order, so every provider
/// gets measured once. After that the enabled provider with the highest throughput wins, where throughput is an
/// exponential moving average of tokens emitted per millisecond. Ties go to the earlier provider.
/// When every provider is disabled, <see cref="Select"/> returns null and the caller decodes without speculation.</para>
/// <para>Auto-disable: a round that proposed tokens counts as below threshold when accepted / proposed is under
/// <see cref="SpeculationSelectorOptions.AcceptanceThreshold"/>. A provider is disabled once that happens for
/// <see cref="SpeculationSelectorOptions.DisableAfterRounds"/> consecutive rounds. A round at or above the threshold
/// resets the streak. Rounds where the provider proposed nothing are neither failures nor successes, so they leave the
/// streak unchanged.</para>
/// </remarks>
public sealed class SpeculationSelector
{
    private readonly SpeculationSelectorOptions _options;
    private readonly List<ProviderState> _ordered = [];
    private readonly Dictionary<string, ProviderState> _byName = new(StringComparer.Ordinal);

    public SpeculationSelector(IEnumerable<ISpeculativeDraftProvider> providers, SpeculationSelectorOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(providers);
        _options = options ?? new SpeculationSelectorOptions();
        if (!double.IsFinite(_options.AcceptanceThreshold) || _options.AcceptanceThreshold < 0 || _options.AcceptanceThreshold > 1)
            throw new ArgumentOutOfRangeException(nameof(options), _options.AcceptanceThreshold, "The acceptance threshold must be between 0 and 1.");
        if (_options.DisableAfterRounds < 1)
            throw new ArgumentOutOfRangeException(nameof(options), _options.DisableAfterRounds, "The disable streak must be at least 1 round.");
        if (!double.IsFinite(_options.ThroughputSmoothing) || _options.ThroughputSmoothing <= 0 || _options.ThroughputSmoothing > 1)
            throw new ArgumentOutOfRangeException(nameof(options), _options.ThroughputSmoothing, "The throughput smoothing must be in (0, 1].");

        foreach (ISpeculativeDraftProvider provider in providers)
        {
            ArgumentNullException.ThrowIfNull(provider);
            ProviderState state = new(provider);
            if (!_byName.TryAdd(provider.Name, state))
                throw new ArgumentException($"Two providers share the name '{provider.Name}'.", nameof(providers));
            _ordered.Add(state);
        }
        if (_ordered.Count == 0) throw new ArgumentException("The selector needs at least one provider.", nameof(providers));
    }

    /// <summary>The provider to draft with this round, or null when every provider has been disabled.</summary>
    public ISpeculativeDraftProvider? Select()
    {
        ISpeculativeDraftProvider? best = null;
        double bestThroughput = double.NegativeInfinity;
        foreach (ProviderState state in _ordered)
        {
            if (state.Disabled) continue;
            if (state.Rounds == 0) return state.Provider;   // measure every enabled provider once before comparing
            double throughput = state.Throughput ?? 0.0;
            if (throughput > bestThroughput)
            {
                best = state.Provider;
                bestThroughput = throughput;
            }
        }
        return best;
    }

    /// <summary>Reports one measured round for <paramref name="provider"/>.</summary>
    public void Record(ISpeculativeDraftProvider provider, SpeculationRound round)
    {
        ArgumentNullException.ThrowIfNull(provider);
        ProviderState state = Lookup(provider);
        if (round.Proposed < 0) throw new ArgumentOutOfRangeException(nameof(round), round.Proposed, "Proposed tokens cannot be negative.");
        if (round.Accepted < 0 || round.Accepted > round.Proposed)
            throw new ArgumentOutOfRangeException(nameof(round), round.Accepted, "Accepted tokens must be between 0 and the proposed count.");
        if (round.TokensEmitted < 0) throw new ArgumentOutOfRangeException(nameof(round), round.TokensEmitted, "Emitted tokens cannot be negative.");
        if (!double.IsFinite(round.ElapsedMilliseconds) || round.ElapsedMilliseconds < 0)
            throw new ArgumentOutOfRangeException(nameof(round), round.ElapsedMilliseconds, "Elapsed time must be finite and non-negative.");

        state.Rounds++;
        if (round.Proposed > 0)
        {
            double acceptance = (double)round.Accepted / round.Proposed;
            state.BelowThresholdStreak = acceptance < _options.AcceptanceThreshold ? state.BelowThresholdStreak + 1 : 0;
            if (state.BelowThresholdStreak >= _options.DisableAfterRounds) state.Disabled = true;
        }
        if (round.ElapsedMilliseconds > 0)
        {
            double sample = round.TokensEmitted / round.ElapsedMilliseconds;
            state.Throughput = state.Throughput is null
                ? sample
                : state.Throughput.Value + _options.ThroughputSmoothing * (sample - state.Throughput.Value);
        }
    }

    /// <summary>Whether <paramref name="provider"/> has been auto-disabled.</summary>
    public bool IsDisabled(ISpeculativeDraftProvider provider) => Lookup(provider).Disabled;

    /// <summary>The moving-average throughput of <paramref name="provider"/> in tokens per millisecond, or null before any timed round.</summary>
    public double? Throughput(ISpeculativeDraftProvider provider) => Lookup(provider).Throughput;

    private ProviderState Lookup(ISpeculativeDraftProvider provider)
    {
        if (!_byName.TryGetValue(provider.Name, out ProviderState? state) || !ReferenceEquals(state.Provider, provider))
            throw new ArgumentException($"The provider '{provider.Name}' is not registered with this selector.", nameof(provider));
        return state;
    }

    private sealed class ProviderState
    {
        public ProviderState(ISpeculativeDraftProvider provider) => Provider = provider;

        public ISpeculativeDraftProvider Provider { get; }
        public int Rounds { get; set; }
        public int BelowThresholdStreak { get; set; }
        public bool Disabled { get; set; }
        public double? Throughput { get; set; }
    }
}
