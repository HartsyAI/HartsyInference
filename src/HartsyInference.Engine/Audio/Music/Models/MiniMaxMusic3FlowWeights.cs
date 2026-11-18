using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Engine.Audio;

/// <summary>The flow transformer (and, when the file carries it, the condition encoder) read from a user-selected checkpoint, with the lifetime of every tensor that borrows or was made from it.</summary>
internal sealed class MiniMaxMusic3FlowWeights : IDisposable
{
    private readonly CheckpointSource _source;
    private readonly List<Tensor> _owned;
    private int _disposed;

    internal MiniMaxMusic3FlowWeights(CheckpointSource source, IReadOnlyDictionary<string, Tensor> transformer,
        IReadOnlyDictionary<string, Tensor>? conditionEncoder, string layout)
    {
        _source = source;
        Transformer = transformer;
        ConditionEncoder = conditionEncoder;
        Layout = layout;
        // A tensor the converter returned that is not one of the file's own was allocated here (a split, a cast, a dequant).
        HashSet<Tensor> borrowed = new HashSet<Tensor>(source.Weights.Values, ReferenceEqualityComparer.Instance as IEqualityComparer<Tensor>);
        _owned = [];
        foreach (IReadOnlyDictionary<string, Tensor>? weights in new[] { transformer, conditionEncoder })
        {
            foreach (Tensor tensor in weights?.Values ?? Enumerable.Empty<Tensor>())
            {
                if (!borrowed.Contains(tensor) && !_owned.Contains(tensor))
                {
                    _owned.Add(tensor);
                }
            }
        }
    }

    internal IReadOnlyDictionary<string, Tensor> Transformer { get; }

    /// <summary>Null when the file holds the transformer only; the condition encoder then comes from the repo.</summary>
    internal IReadOnlyDictionary<string, Tensor>? ConditionEncoder { get; }

    /// <summary>Human-readable layout, for the load log.</summary>
    internal string Layout { get; }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        foreach (Tensor tensor in _owned)
        {
            tensor.Dispose();
        }
        _source.Dispose();
    }
}
