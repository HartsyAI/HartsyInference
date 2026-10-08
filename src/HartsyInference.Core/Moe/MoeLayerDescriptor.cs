using System.Collections.ObjectModel;

namespace HartsyInference.Core.Moe;

/// <summary>
/// A routed expert group: how many experts, their shape, and optional per-index shape overrides. The overrides are copied
/// at construction, so a caller cannot change a group after the topology fingerprint was computed.
/// </summary>
public sealed record ExpertGroupDescriptor
{
    /// <summary>Creates a group.</summary>
    /// <param name="count">Experts in the group.</param>
    /// <param name="shape">Shape of every expert not listed in <paramref name="overrides"/>.</param>
    /// <param name="overrides">Experts whose shape differs from <paramref name="shape"/>, keyed by index; null when uniform.</param>
    /// <exception cref="ArgumentException">An override index is outside the group, or a value is null.</exception>
    public ExpertGroupDescriptor(int count, ExpertDescriptor shape, IReadOnlyDictionary<int, ExpertDescriptor>? overrides = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        ArgumentNullException.ThrowIfNull(shape);
        Count = count;
        Shape = shape;
        if (overrides is null)
        {
            Overrides = null;
            return;
        }
        Dictionary<int, ExpertDescriptor> copy = new(overrides.Count);
        foreach (KeyValuePair<int, ExpertDescriptor> pair in overrides)
        {
            if ((uint)pair.Key >= (uint)count) throw new ArgumentException($"Override index {pair.Key} is outside a group of {count}.", nameof(overrides));
            ArgumentNullException.ThrowIfNull(pair.Value);
            copy[pair.Key] = pair.Value;
        }
        Overrides = new ReadOnlyDictionary<int, ExpertDescriptor>(copy);
    }

    /// <summary>Experts in the group.</summary>
    public int Count { get; }

    /// <summary>Shape of every expert without an override.</summary>
    public ExpertDescriptor Shape { get; }

    /// <summary>Per-index shapes that differ from <see cref="Shape"/>, or null when the group is uniform.</summary>
    public IReadOnlyDictionary<int, ExpertDescriptor>? Overrides { get; }

    /// <summary>Kept for call-site symmetry; validation happens in the constructor.</summary>
    public ExpertGroupDescriptor Validated() => this;

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
    /// <exception cref="ArgumentException">Router width and expert count disagree.</exception>
    public MoeLayerDescriptor Validated()
    {
        ArgumentNullException.ThrowIfNull(Router);
        ArgumentNullException.ThrowIfNull(Routed);
        ArgumentNullException.ThrowIfNull(Program);
        Router.Validated();
        Routed.Validated();
        Shared?.Validated();
        Program.Validated();
        if (Router.NumExperts != Routed.Count)
            throw new ArgumentException($"Router selects among {Router.NumExperts} experts but the routed group holds {Routed.Count}.", nameof(Router));
        return this;
    }

    /// <summary>Number of routed experts.</summary>
    public int ExpertCount => Routed.Count;

    /// <summary>Payload bytes of all routed and shared experts in this layer.</summary>
    public long PayloadBytes => Routed.PayloadBytes + (Shared?.PayloadBytes ?? 0);
}
