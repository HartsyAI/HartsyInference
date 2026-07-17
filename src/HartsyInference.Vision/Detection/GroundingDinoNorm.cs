using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>A single learned LayerNorm applied at a residual junction. Grounding DINO uses post-norm
/// (<c>LayerNorm(x + sublayer(x))</c>); this owns the norm weight/bias for one such junction and offers a fused
/// "add residual then normalize" helper so the many residual points across the enhancer and decoder do not each
/// re-implement it.</summary>
public sealed class GroundingDinoNorm
{
    private const float Eps = 1e-5f;

    private readonly int _dim;
    private Tensor? _weight, _bias;

    public GroundingDinoNorm(int dim)
    {
        _dim = dim;
    }

    public void Build(IGroundingWeightBag bag, string prefix)
    {
        _weight = bag.Get($"{prefix}.weight", _dim);
        _bias = bag.Get($"{prefix}.bias", _dim);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_weight is not null) yield return _weight;
        if (_bias is not null) yield return _bias;
    }

    /// <summary>Post-norm residual: returns <c>LayerNorm(residual + sublayer)</c>. Disposes <paramref name="sublayer"/>
    /// (an owned intermediate); <paramref name="residual"/> is left untouched for the caller to dispose.</summary>
    public Tensor AddNorm(IBackend backend, Tensor residual, Tensor sublayer)
    {
        if (_weight is null)
            throw new InvalidOperationException("GroundingDinoNorm weights not loaded.");
        backend.Add(sublayer, sublayer, residual);
        Tensor output = new Tensor(sublayer.Shape, DType.F32);
        backend.LayerNorm(output, sublayer, _weight!, _bias!, Eps);
        sublayer.Dispose();
        return output;
    }
}
