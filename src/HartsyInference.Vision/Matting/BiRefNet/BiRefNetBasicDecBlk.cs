using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Annotators;

namespace HartsyInference.Vision.Matting.BiRefNet;

/// <summary>Upstream <c>BasicDecBlk</c> in eval mode: 3×3 conv → BN → ReLU → deformable ASPP (64 channels) → 3×3 conv → BN.
/// Both BatchNorms are folded into their convs at load.</summary>
internal sealed unsafe class BiRefNetBasicDecBlk
{
    private const float BatchNormEps = 1e-5f;

    private readonly BiRefNetAspp _aspp = new();
    private Tensor? _inW, _inB, _outW, _outB;

    /// <summary>Loads <c>{prefix}.conv_in/bn_in</c>, <c>{prefix}.dec_att.*</c> and <c>{prefix}.conv_out/bn_out</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, List<Tensor> owned)
    {
        (Tensor inW, Tensor inB) = ConvBnFold.Fold(w, $"{prefix}.conv_in", $"{prefix}.bn_in", BatchNormEps);
        (Tensor outW, Tensor outB) = ConvBnFold.Fold(w, $"{prefix}.conv_out", $"{prefix}.bn_out", BatchNormEps);
        owned.Add(inW); owned.Add(inB); owned.Add(outW); owned.Add(outB);
        _inW = inW; _inB = inB; _outW = outW; _outB = outB;
        _aspp.LoadWeights(w, $"{prefix}.dec_att", owned);
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        Tensor convIn = BiRefNetOps.Conv(backend, x, _inW!, _inB, pad: 1);
        Tensor act = BiRefNetOps.Relu(backend, convIn);
        convIn.Dispose();
        Tensor attended = _aspp.Forward(backend, act);
        act.Dispose();
        Tensor output = BiRefNetOps.Conv(backend, attended, _outW!, _outB, pad: 1);
        attended.Dispose();
        return output;
    }
}
