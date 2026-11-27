using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Utonia;

/// <summary>Dense layer over <c>[rows, in]</c> features: <c>y = x Wᵀ + b</c> with PyTorch's <c>[out, in]</c> weight layout.</summary>
internal sealed class UtoniaLinear
{
    private readonly IBackend _backend;
    private readonly Tensor _weight;
    private readonly Tensor? _bias;

    public UtoniaLinear(IBackend backend, Tensor weight, Tensor? bias)
    {
        _backend = backend;
        _weight = weight;
        _bias = bias;
        OutChannels = (int)weight.Shape[0];
    }

    public int OutChannels { get; }

    public Tensor Forward(Tensor input)
    {
        Tensor output = new(new TensorShape(input.Shape[0], OutChannels), DType.F32);
        Forward(input, output);
        return output;
    }

    /// <summary>Writes <c>[rows, OutChannels]</c> into <paramref name="output"/>; used per row chunk.</summary>
    public void Forward(Tensor input, Tensor output) => _backend.Linear(output, input, _weight, _bias);
}
