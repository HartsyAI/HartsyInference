using HartsyInference.Core.Tensors;
namespace HartsyInference.Core.Backends;

/// <summary>The three projections of one expert. Its tensor objects are the cache identity, so build it once per expert (see <see cref="ExpertBank"/>).</summary>
public sealed class ExpertWeights
{
    /// <summary>Creates the weights of <paramref name="key"/>; W1 and W3 are the gate and up projections, W2 the down projection.</summary>
    public ExpertWeights(ExpertKey key, ExpertMatrix w1, ExpertMatrix w2, ExpertMatrix w3)
    {
        Key = key;
        W1 = w1 ?? throw new ArgumentNullException(nameof(w1));
        W2 = w2 ?? throw new ArgumentNullException(nameof(w2));
        W3 = w3 ?? throw new ArgumentNullException(nameof(w3));
        List<Tensor> tensors = new(9);
        foreach (ExpertMatrix matrix in new[] { w1, w2, w3 })
        {
            AddDistinct(tensors, matrix.Weight);
            foreach (Tensor companion in matrix.Companions) AddDistinct(tensors, companion);
        }
        Tensors = tensors;
        long bytes = 0;
        foreach (Tensor tensor in tensors) bytes += tensor.DType.ComputeByteCount(tensor.ElementCount);
        Bytes = bytes;
    }

    /// <summary>The expert these weights belong to.</summary>
    public ExpertKey Key { get; }

    /// <summary>Gate projection.</summary>
    public ExpertMatrix W1 { get; }

    /// <summary>Down projection.</summary>
    public ExpertMatrix W2 { get; }

    /// <summary>Up projection.</summary>
    public ExpertMatrix W3 { get; }

    /// <summary>Every tensor that has to be resident for this expert to run: three weights and their scales, global scales and biases.</summary>
    public IReadOnlyList<Tensor> Tensors { get; }

    /// <summary>Total device bytes of <see cref="Tensors"/>.</summary>
    public long Bytes { get; }

    private static void AddDistinct(List<Tensor> tensors, Tensor tensor)
    {
        foreach (Tensor existing in tensors)
        {
            if (ReferenceEquals(existing, tensor)) return;
        }
        tensors.Add(tensor);
    }
}
