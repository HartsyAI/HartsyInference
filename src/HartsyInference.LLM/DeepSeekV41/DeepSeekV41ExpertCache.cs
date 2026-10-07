namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Least-recently-used cache of dequantized experts in front of a loader, so hot experts are dequantized once.</summary>
/// <remarks>Not thread-safe; the CPU reference path runs one layer at a time. Capacity is in experts, not bytes.</remarks>
public sealed class DeepSeekV41ExpertCache : IDeepSeekV41ExpertSource
{
    private readonly Func<int, DeepSeekV41SwigluWeights> _load;
    private readonly int _capacity;
    private readonly Dictionary<int, LinkedListNode<(int Expert, DeepSeekV41SwigluWeights Weights)>> _index = [];
    private readonly LinkedList<(int Expert, DeepSeekV41SwigluWeights Weights)> _order = new();

    /// <summary>Creates a cache holding at most <paramref name="capacity"/> experts.</summary>
    public DeepSeekV41ExpertCache(Func<int, DeepSeekV41SwigluWeights> load, int capacity)
    {
        ArgumentNullException.ThrowIfNull(load);
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        _load = load;
        _capacity = capacity;
    }

    /// <summary>Experts currently cached.</summary>
    public int Count => _index.Count;

    /// <inheritdoc />
    public DeepSeekV41SwigluWeights GetExpert(int expert)
    {
        if (_index.TryGetValue(expert, out LinkedListNode<(int Expert, DeepSeekV41SwigluWeights Weights)>? hit))
        {
            _order.Remove(hit);
            _order.AddFirst(hit);
            return hit.Value.Weights;
        }
        DeepSeekV41SwigluWeights loaded = _load(expert);
        if (_index.Count >= _capacity)
        {
            LinkedListNode<(int Expert, DeepSeekV41SwigluWeights Weights)> oldest = _order.Last!;
            _order.RemoveLast();
            _index.Remove(oldest.Value.Expert);
        }
        _index[expert] = _order.AddFirst((expert, loaded));
        return loaded;
    }
}
