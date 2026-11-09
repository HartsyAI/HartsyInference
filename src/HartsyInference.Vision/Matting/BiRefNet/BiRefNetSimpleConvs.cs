using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Matting.BiRefNet;

/// <summary>Upstream <c>SimpleConvs</c> (the image-patch injection blocks): 3×3 conv to 64 channels, then 3×3 conv to the
/// output width, with no normalization or activation between them.</summary>
internal sealed unsafe class BiRefNetSimpleConvs
{
    private Tensor? _w1, _b1, _wOut, _bOut;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, List<Tensor> owned)
    {
        _w1 = BiRefNetOps.Own(w[$"{prefix}.conv1.weight"], owned);
        _b1 = BiRefNetOps.Own(w[$"{prefix}.conv1.bias"], owned);
        _wOut = BiRefNetOps.Own(w[$"{prefix}.conv_out.weight"], owned);
        _bOut = BiRefNetOps.Own(w[$"{prefix}.conv_out.bias"], owned);
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        Tensor mid = BiRefNetOps.Conv(backend, x, _w1!, _b1, pad: 1);
        Tensor output = BiRefNetOps.Conv(backend, mid, _wOut!, _bOut, pad: 1);
        mid.Dispose();
        return output;
    }
}
