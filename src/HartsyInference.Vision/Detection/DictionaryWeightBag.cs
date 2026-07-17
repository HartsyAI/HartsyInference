using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>Resolves Grounding DINO parameters from a loaded safetensors dictionary, casting to F32 and validating
/// that the stored element count matches the shape the module asked for (a mismatched key silently reshaping a
/// weight is the classic cause of garbage output, so we fail fast instead).</summary>
public sealed class DictionaryWeightBag : IGroundingWeightBag
{
    private readonly IReadOnlyDictionary<string, Tensor> _weights;

    public DictionaryWeightBag(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _weights = weights;
    }

    public Tensor Get(string name, params int[] dims)
    {
        if (!_weights.TryGetValue(name, out Tensor? tensor))
            throw new KeyNotFoundException($"Grounding DINO weight '{name}' not found in checkpoint.");

        long expected = 1;
        for (int i = 0; i < dims.Length; i++)
            expected *= dims[i];
        if (tensor.Shape.ElementCount != expected)
            throw new ArgumentException($"Weight '{name}' has {tensor.Shape.ElementCount} elements; module expected {expected} ({string.Join("x", dims)}).");

        Tensor f32 = tensor.DType != DType.F32 ? tensor.CastTo(DType.F32) : tensor;
        return f32.Shape.Rank == dims.Length ? f32 : f32.Reshape(new TensorShape(ToLong(dims)));
    }

    private static long[] ToLong(int[] dims)
    {
        long[] result = new long[dims.Length];
        for (int i = 0; i < dims.Length; i++)
            result[i] = dims[i];
        return result;
    }
}
