using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>Affine LayerNorm over the last dimension. Holds the <c>[dim]</c> weight + bias and calls
/// <see cref="IBackend.LayerNorm"/>; used by every RT-DETR transformer sub-block (post-norm layout).</summary>
public sealed class RtDetrLayerNorm
{
    private readonly int _dim;
    private readonly float _eps;

    private Tensor? _weight;   // [dim]
    private Tensor? _bias;     // [dim]

    /// <summary>Normalized feature dimension.</summary>
    public int Dim => _dim;

    /// <summary>Creates a LayerNorm over <paramref name="dim"/> features.</summary>
    public RtDetrLayerNorm(int dim, float eps)
    {
        if (dim <= 0)
            throw new ArgumentOutOfRangeException(nameof(dim), dim, "Dim must be positive.");
        _dim = dim;
        _eps = eps;
    }

    /// <summary>Loads <c>{prefix}.weight</c> and <c>{prefix}.bias</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _weight = EnsureF32(weights[$"{prefix}.weight"]);
        _bias = EnsureF32(weights[$"{prefix}.bias"]);
    }

    /// <summary>Yields the weight + bias for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_weight is not null) yield return _weight;
        if (_bias is not null) yield return _bias;
    }

    /// <summary>Normalizes <paramref name="input"/> in place (output may alias input). Owns the returned tensor when it allocates.</summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        if (_weight is null || _bias is null)
            throw new InvalidOperationException("RtDetrLayerNorm weights not loaded.");
        Tensor output = new Tensor(input.Shape, DType.F32);
        backend.LayerNorm(output, input, _weight, _bias, _eps);
        return output;
    }

    private static Tensor EnsureF32(Tensor t) =>
        t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}
