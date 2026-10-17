using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Looped;

/// <summary>The constant unit-norm weights a looped normalized-GPT decoder normalizes with, one per width it normalizes over
/// (<see cref="HypersphereOps.InverseSqrtVector"/>), shared by every layer and uploaded with the weights.</summary>
internal sealed class LoopedNormVectors : IDisposable
{
    private int _disposed;

    /// <summary>Residual stream width.</summary>
    public Tensor Hidden { get; }

    /// <summary>One attention head.</summary>
    public Tensor Head { get; }

    /// <summary>The merged attention output, all query heads.</summary>
    public Tensor Attention { get; }

    /// <summary>The MLP inner width.</summary>
    public Tensor Ffn { get; }

    public LoopedNormVectors(LoopedTransformerConfig cfg)
    {
        Hidden = HypersphereOps.InverseSqrtVector(cfg.HiddenSize);
        Head = HypersphereOps.InverseSqrtVector(cfg.HeadDim);
        Attention = HypersphereOps.InverseSqrtVector(cfg.QDim);
        Ffn = HypersphereOps.InverseSqrtVector(cfg.IntermediateSize);
    }

    /// <summary>The four vectors, for residency.</summary>
    public IEnumerable<Tensor> Enumerate()
    {
        yield return Hidden;
        yield return Head;
        yield return Attention;
        yield return Ffn;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Hidden.Dispose();
        Head.Dispose();
        Attention.Dispose();
        Ffn.Dispose();
    }
}
