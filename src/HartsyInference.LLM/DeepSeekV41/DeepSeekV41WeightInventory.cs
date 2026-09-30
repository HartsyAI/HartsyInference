using HartsyInference.Core.Exceptions;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Byte and tensor totals of a DeepSeek-V4.1 checkpoint per memory class, summed from headers alone.</summary>
public sealed record DeepSeekV41WeightInventory
{
    private const int MaxListedUnknown = 8;

    /// <summary>Tensor bytes per class; every class is present, zero when the checkpoint has none.</summary>
    public required IReadOnlyDictionary<DeepSeekV41WeightClass, long> BytesByClass { get; init; }

    /// <summary>Tensor count per class; every class is present.</summary>
    public required IReadOnlyDictionary<DeepSeekV41WeightClass, int> CountsByClass { get; init; }

    /// <summary>Sum of every class; equals the index's <c>metadata.total_size</c> for a complete official checkpoint.</summary>
    public long TotalBytes => BytesByClass.Values.Sum();

    /// <summary>Total tensors across all classes.</summary>
    public int TotalCount => CountsByClass.Values.Sum();

    /// <summary>Sums <paramref name="tensors"/> (canonical key, byte length) per class.</summary>
    /// <exception cref="HartsyInferenceException">A key belongs to no class, so the totals would silently drop it.</exception>
    public static DeepSeekV41WeightInventory Summarize(IEnumerable<KeyValuePair<string, long>> tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        Dictionary<DeepSeekV41WeightClass, long> bytes = new();
        Dictionary<DeepSeekV41WeightClass, int> counts = new();
        foreach (DeepSeekV41WeightClass weightClass in Enum.GetValues<DeepSeekV41WeightClass>())
        {
            bytes[weightClass] = 0;
            counts[weightClass] = 0;
        }
        List<string> unknown = new();
        foreach (KeyValuePair<string, long> tensor in tensors)
        {
            DeepSeekV41WeightClass? weightClass = DeepSeekV41WeightClassifier.Classify(tensor.Key);
            if (weightClass is null)
            {
                unknown.Add(tensor.Key);
                continue;
            }
            bytes[weightClass.Value] += tensor.Value;
            counts[weightClass.Value]++;
        }
        if (unknown.Count > 0)
        {
            string listed = string.Join(", ", unknown.Order(StringComparer.Ordinal).Take(MaxListedUnknown));
            throw new HartsyInferenceException(
                $"{unknown.Count} tensors match no DeepSeek-V4.1 weight class (first: {listed}); the memory estimate would drop them.");
        }
        return new DeepSeekV41WeightInventory { BytesByClass = bytes, CountsByClass = counts };
    }
}
