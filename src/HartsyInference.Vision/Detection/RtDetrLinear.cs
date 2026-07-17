using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>A single fully-connected layer (<c>output = input · weightᵀ + bias</c>) with weight
/// <c>[out, in]</c> and optional bias <c>[out]</c>. Thin wrapper over <see cref="IBackend.Linear"/>
/// that owns weight loading + preload enumeration so the RT-DETR transformer blocks don't repeat
/// the same three lines for every projection.</summary>
public sealed class RtDetrLinear
{
    private readonly int _outFeatures;
    private readonly bool _hasBias;

    private Tensor? _weight;   // [out, in]
    private Tensor? _bias;     // [out]

    /// <summary>Output feature count.</summary>
    public int OutFeatures => _outFeatures;

    /// <summary>Creates a linear layer. <paramref name="hasBias"/> false skips the bias tensor entirely (e.g. an attention QKV projection trained without bias).</summary>
    public RtDetrLinear(int outFeatures, bool hasBias = true)
    {
        if (outFeatures <= 0)
            throw new ArgumentOutOfRangeException(nameof(outFeatures), outFeatures, "Out features must be positive.");
        _outFeatures = outFeatures;
        _hasBias = hasBias;
    }

    /// <summary>Loads <c>{prefix}.weight</c> (and <c>{prefix}.bias</c> when this layer has a bias).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _weight = EnsureF32(weights[$"{prefix}.weight"]);
        if (_hasBias)
            _bias = EnsureF32(weights[$"{prefix}.bias"]);
    }

    /// <summary>Yields the weight (and bias) for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_weight is not null) yield return _weight;
        if (_bias is not null) yield return _bias;
    }

    /// <summary>Runs the projection on <paramref name="input"/> <c>[..., in]</c> → <c>[..., out]</c>. Owns the returned tensor.</summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        if (_weight is null)
            throw new InvalidOperationException("RtDetrLinear weights not loaded.");

        int rank = input.Shape.Rank;
        Span<long> dims = stackalloc long[rank];
        for (int i = 0; i < rank - 1; i++)
            dims[i] = input.Shape[i];
        dims[rank - 1] = _outFeatures;

        Tensor output = new Tensor(new TensorShape(dims), DType.F32);
        backend.Linear(output, input, _weight, _bias);
        return output;
    }

    private static Tensor EnsureF32(Tensor t) =>
        t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}
