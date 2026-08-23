using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.ThreeD.Models.Trellis;

/// <summary>TRELLIS image conditioner: the <c>dinov2_vitl14_reg</c> backbone tapped at <c>x_prenorm</c> (pre-final-norm), then a non-affine LayerNorm producing the <c>[1, 1374, 1024]</c> conditioning tokens both flow stages cross-attend to.</summary>
public sealed class TrellisImageConditioner
{
    private readonly Dinov2VisionEncoder _dino = new(Dinov2Preset.LargeReg);

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights) => _dino.LoadWeights(weights);
    public IEnumerable<Tensor> EnumerateWeights() => _dino.EnumerateWeights();

    /// <summary>ImageNet-normalized pixels <c>[1,3,518,518]</c> → cond tokens <c>[1,1374,1024]</c>.</summary>
    public Tensor Encode(IBackend backend, Tensor pixelValues)
    {
        Tensor pre = _dino.Encode(backend, pixelValues, applyFinalNorm: false);   // x_prenorm tap
        Tensor cond = new(pre.Shape, DType.F32);
        backend.LayerNormNoAffine(cond, pre, 1e-5f);                              // F.layer_norm(·, [1024])
        pre.Dispose();
        return cond;
    }
}
