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
