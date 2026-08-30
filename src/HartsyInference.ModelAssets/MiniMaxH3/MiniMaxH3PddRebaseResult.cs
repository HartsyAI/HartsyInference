using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;

namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>Non-AdaLN LoRA layers plus owned curve-coordinate weight/DC-bias diffs.</summary>
public sealed class MiniMaxH3PddRebaseResult : IDisposable
{
    private readonly IReadOnlyList<Tensor> _ownedTensors;
    private int _disposed;

    /// <summary>Creates an owned rebase result.</summary>
    public MiniMaxH3PddRebaseResult(IReadOnlyList<LoraLayer> layers,
        IReadOnlyList<LoraFullWeightDiff> fullWeightDiffs, IReadOnlyList<Tensor> ownedTensors)
    {
        Layers = layers ?? throw new ArgumentNullException(nameof(layers));
        FullWeightDiffs = fullWeightDiffs ?? throw new ArgumentNullException(nameof(fullWeightDiffs));
        _ownedTensors = ownedTensors ?? throw new ArgumentNullException(nameof(ownedTensors));
    }

    /// <summary>Trunk updates that remain ordinary low-rank matrices.</summary>
    public IReadOnlyList<LoraLayer> Layers { get; }

    /// <summary>Paired curve-weight and mandatory DC-bias diffs for every dense AdaLN target.</summary>
    public IReadOnlyList<LoraFullWeightDiff> FullWeightDiffs { get; }

    /// <summary>Releases the generated F32 diffs.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (Tensor tensor in _ownedTensors) tensor.Dispose();
    }
}
