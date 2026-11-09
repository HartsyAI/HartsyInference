using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>LayerNorm over the last dimension with PyTorch's default epsilon.</summary>
internal sealed class UtoniaLayerNorm
{
    private const float Epsilon = 1e-5f;

    private readonly IBackend _backend;
    private readonly Tensor _weight;
    private readonly Tensor _bias;

    public UtoniaLayerNorm(IBackend backend, Tensor weight, Tensor bias)
    {
        _backend = backend;
        _weight = weight;
        _bias = bias;
    }

    public Tensor Forward(Tensor input)
    {
        Tensor output = new(input.Shape, DType.F32);
        _backend.LayerNorm(output, input, _weight, _bias, Epsilon);
        return output;
    }
}
