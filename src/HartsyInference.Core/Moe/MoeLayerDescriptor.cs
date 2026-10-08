namespace HartsyInference.Core.Moe;

/// <summary>A routed expert group: how many experts, their shape, and optional per-index shape overrides.</summary>
/// <param name="Count">Experts in the group.</param>
/// <param name="Shape">Shape of every expert not listed in <paramref name="Overrides"/>.</param>
/// <param name="Overrides">Experts whose shape differs from <paramref name="Shape"/>, keyed by index; null when uniform.</param>
public sealed record ExpertGroupDescriptor(int Count, ExpertDescriptor Shape, IReadOnlyDictionary<int, ExpertDescriptor>? Overrides = null)
{
    /// <summary>Validates counts and override indices.</summary>
    /// <exception cref="ArgumentException">An override is outside the group.</exception>
    public ExpertGroupDescriptor Validated()
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(Count);
        ArgumentNullException.ThrowIfNull(Shape);
        if (Overrides is not null)
        {
            foreach (int index in Overrides.Keys)
                if ((uint)index >= (uint)Count) throw new ArgumentException($"Override index {index} is outside a group of {Count}.", nameof(Overrides));
        }
        return this;
    }

    /// <summary>The shape of expert <paramref name="index"/>.</summary>
    public ExpertDescriptor ExpertAt(int index)
    {
        if ((uint)index >= (uint)Count) throw new ArgumentOutOfRangeException(nameof(index), index, $"Group has {Count} experts.");
        return Overrides is not null && Overrides.TryGetValue(index, out ExpertDescriptor? shape) ? shape : Shape;
    }

    /// <summary>Total payload bytes across the group.</summary>
    public long PayloadBytes
    {
        get
        {
            long total = (long)Count * Shape.PayloadBytes;
            if (Overrides is not null)
                foreach (KeyValuePair<int, ExpertDescriptor> pair in Overrides) total += pair.Value.PayloadBytes - Shape.PayloadBytes;
            return total;
        }
    }
}

/// <summary>
/// One sparse (MoE) feed-forward layer: a router selecting among routed experts, an optional shared expert group that
/// always runs, and the expert program both groups execute.
/// </summary>
/// <param name="Router">Routing recipe; its <see cref="RouterDescriptor.NumExperts"/> must equal <see cref="Routed"/>.<see cref="ExpertGroupDescriptor.Count"/>.</param>
/// <param name="Routed">Routed experts.</param>
/// <param name="Shared">Always-on shared experts, or null.</param>
/// <param name="SharedIsGated">Shared output is scaled by a learned sigmoid gate (Qwen2-MoE).</param>
/// <param name="Program">Computation each expert runs.</param>
public sealed record MoeLayerDescriptor(
    RouterDescriptor Router, ExpertGroupDescriptor Routed, ExpertGroupDescriptor? Shared, bool SharedIsGated, ExpertProgram Program)
{
    /// <summary>Validates the layer and returns it.</summary>
    /// <exception cref="ArgumentException">Router width and expert count disagree, or the shared group is invalid.</exception>
    public MoeLayerDescriptor Validated()
    {
        ArgumentNullException.ThrowIfNull(Router);
        ArgumentNullException.ThrowIfNull(Routed);
        ArgumentNullException.ThrowIfNull(Program);
        Router.Validated();
        Routed.Validated();
        Shared?.Validated();
        if (Router.NumExperts != Routed.Count)
            throw new ArgumentException($"Router selects among {Router.NumExperts} experts but the routed group holds {Routed.Count}.", nameof(Router));
        return this;
    }

    /// <summary>Number of routed experts.</summary>
    public int ExpertCount => Routed.Count;

    /// <summary>Payload bytes of all routed and shared experts in this layer.</summary>
    public long PayloadBytes => Routed.PayloadBytes + (Shared?.PayloadBytes ?? 0);
}
