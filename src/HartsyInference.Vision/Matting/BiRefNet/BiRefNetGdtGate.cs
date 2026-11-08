using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Annotators;

namespace HartsyInference.Vision.Matting.BiRefNet;

/// <summary>Upstream gradient-guided gate (<c>gdt_convs_N</c> + <c>gdt_convs_attn_N</c>) applied at decoder levels 4–2 in
/// eval mode: a 3×3 conv → BN → ReLU to 16 channels, a 1×1 conv to one channel, sigmoid, and the decoder feature is
/// multiplied by that single-channel gate.</summary>
internal sealed unsafe class BiRefNetGdtGate
{
    private const float BatchNormEps = 1e-5f;

    private Tensor? _convW, _convB, _attnW, _attnB;

    /// <summary>Loads <c>{convPrefix}.0</c> with its BN at <c>{convPrefix}.1</c>, and the gate conv at <c>{attnPrefix}.0</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string convPrefix, string attnPrefix, List<Tensor> owned)
    {
        (Tensor convW, Tensor convB) = ConvBnFold.Fold(w, $"{convPrefix}.0", $"{convPrefix}.1", BatchNormEps);
        owned.Add(convW); owned.Add(convB);
        _convW = convW;
        _convB = convB;
        _attnW = BiRefNetOps.Own(w[$"{attnPrefix}.0.weight"], owned);
        _attnB = BiRefNetOps.Own(w[$"{attnPrefix}.0.bias"], owned);
    }

    /// <summary>Returns <c>x · sigmoid(attn(relu(conv(x))))</c> as a fresh tensor; <paramref name="x"/> is left unchanged.</summary>
    public Tensor Forward(IBackend backend, Tensor x)
    {
        Tensor hidden = BiRefNetOps.Conv(backend, x, _convW!, _convB, pad: 1);
        Tensor act = BiRefNetOps.Relu(backend, hidden);
        hidden.Dispose();
        Tensor logits = BiRefNetOps.Conv(backend, act, _attnW!, _attnB, pad: 0);
        act.Dispose();
        Tensor gate = BiRefNetOps.Sigmoid(backend, logits);
        logits.Dispose();
        Tensor gated = BiRefNetOps.MulBroadcastGate(x, gate);
        gate.Dispose();
        return gated;
    }
}
