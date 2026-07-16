using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.ThreeD.Models.Trellis;

/// <summary>TRELLIS image conditioner: the <c>dinov2_vitl14_reg</c> backbone tapped at <c>x_prenorm</c> (block
/// output before the final norm), then a <b>non-affine</b> LayerNorm over the last dim → <c>[1, 1374, 1024]</c>
/// conditioning tokens (1 CLS + 4 registers + 37² patches) that both flow stages cross-attend to. Mirrors the
/// reference exactly: <c>dino(t, is_training=True)['x_prenorm']</c> then <c>F.layer_norm(feats, feats.shape[-1:])</c>
/// (torch default eps 1e-5). Weights = the torch.hub checkpoint remapped to HF keys (<c>convert_dinov2_reg.py</c>).</summary>
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
