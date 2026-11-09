using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>
/// Named Utonia parameters keyed by C# name (see <see cref="UtoniaCheckpointKeys"/>). Records every name the encoder
/// requests, so a test can check the requested set against the set the checkpoint key map implies.
/// </summary>
public sealed class UtoniaWeights
{
    private readonly IReadOnlyDictionary<string, Tensor> _tensors;
    private readonly HashSet<string> _consumed = new(StringComparer.Ordinal);

    public UtoniaWeights(IReadOnlyDictionary<string, Tensor> tensors)
    {
        _tensors = tensors ?? throw new ArgumentNullException(nameof(tensors));
    }

    /// <summary>Names the encoder has requested so far.</summary>
    public IReadOnlySet<string> Consumed => _consumed;

    /// <summary>Every name supplied.</summary>
    public IEnumerable<string> Provided => _tensors.Keys;

    /// <summary>Returns the named tensor after checking its dtype (F32 only) and exact shape.</summary>
    internal Tensor Get(string name, params long[] shape)
    {
        if (!_tensors.TryGetValue(name, out Tensor? tensor))
        {
            throw new HartsyInferenceException($"Utonia weight '{name}' is missing.");
        }

        if (tensor.DType != DType.F32)
        {
            throw new HartsyInferenceException($"Utonia weight '{name}' has dtype {tensor.DType.Name}, expected F32.");
        }

        TensorShape expected = new(shape);
        if (!tensor.Shape.Equals(expected))
        {
            throw new HartsyInferenceException($"Utonia weight '{name}' has shape {tensor.Shape}, expected {expected}.");
        }

        _consumed.Add(name);
        return tensor;
    }
}
