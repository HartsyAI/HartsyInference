using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection.Blocks;

namespace HartsyInference.Vision.Detection;

/// <summary>RT-DETR CNN backbone — a CSP-style feature extractor assembled from the shared
/// <see cref="ConvBnSilu"/> / <see cref="C2f"/> blocks. It reproduces the multi-scale output
/// <i>contract</i> of the reference ResNet-50 backbone (three feature maps at strides 8/16/32,
/// channels C3/C4/C5) rather than the exact ResNet block topology — the point B5 target is a
/// shape-correct, weight-loadable forward, not ResNet parity.
/// <para>Stem downsamples ×4 (two stride-2 convs); each of the three stages then downsamples ×2
/// (stride-2 conv) and refines with a C2f, yielding C3 @ /8, C4 @ /16, C5 @ /32.</para></summary>
public sealed class RtDetrBackbone
{
    private readonly ConvBnSilu _stem0;   // 3 -> stem0, stride 2  (=> /2)
    private readonly ConvBnSilu _stem1;   // stem0 -> stem1, stride 2  (=> /4)

    private readonly ConvBnSilu _down3;   // stem1 -> C3, stride 2  (=> /8)
    private readonly C2f _c2f3;
    private readonly ConvBnSilu _down4;   // C3 -> C4, stride 2  (=> /16)
    private readonly C2f _c2f4;
    private readonly ConvBnSilu _down5;   // C4 -> C5, stride 2  (=> /32)
    private readonly C2f _c2f5;

    /// <summary>Builds the backbone from a config's <see cref="RtDetrConfig.BackboneChannels"/>.</summary>
    public RtDetrBackbone(RtDetrConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        int stem0 = config.BackboneChannels[0];
        int stem1 = config.BackboneChannels[1];
        int c3 = config.BackboneChannels[2];
        int c4 = config.BackboneChannels[3];
        int c5 = config.BackboneChannels[4];
        int n = config.BackboneRepeat;

        _stem0 = new ConvBnSilu(stem0, strideH: 2, strideW: 2, padH: 1, padW: 1);
        _stem1 = new ConvBnSilu(stem1, strideH: 2, strideW: 2, padH: 1, padW: 1);

        _down3 = new ConvBnSilu(c3, strideH: 2, strideW: 2, padH: 1, padW: 1);
        _c2f3 = new C2f(c3, c3, n, shortcut: true);
        _down4 = new ConvBnSilu(c4, strideH: 2, strideW: 2, padH: 1, padW: 1);
        _c2f4 = new C2f(c4, c4, n, shortcut: true);
        _down5 = new ConvBnSilu(c5, strideH: 2, strideW: 2, padH: 1, padW: 1);
        _c2f5 = new C2f(c5, c5, n, shortcut: true);
    }

    /// <summary>Loads backbone weights under <paramref name="prefix"/> (default <c>"backbone"</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _stem0.LoadWeights(weights, $"{prefix}.stem.0.conv");
        _stem1.LoadWeights(weights, $"{prefix}.stem.1.conv");
        _down3.LoadWeights(weights, $"{prefix}.stage3.downsample.conv");
        _c2f3.LoadWeights(weights, $"{prefix}.stage3.c2f");
        _down4.LoadWeights(weights, $"{prefix}.stage4.downsample.conv");
        _c2f4.LoadWeights(weights, $"{prefix}.stage4.c2f");
        _down5.LoadWeights(weights, $"{prefix}.stage5.downsample.conv");
        _c2f5.LoadWeights(weights, $"{prefix}.stage5.c2f");
    }

    /// <summary>Yields every backbone weight for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _stem0.EnumerateWeights()) yield return t;
        foreach (Tensor t in _stem1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _down3.EnumerateWeights()) yield return t;
        foreach (Tensor t in _c2f3.EnumerateWeights()) yield return t;
        foreach (Tensor t in _down4.EnumerateWeights()) yield return t;
        foreach (Tensor t in _c2f4.EnumerateWeights()) yield return t;
        foreach (Tensor t in _down5.EnumerateWeights()) yield return t;
        foreach (Tensor t in _c2f5.EnumerateWeights()) yield return t;
    }

    /// <summary>Runs the backbone. Returns the three feature maps <c>(C3, C4, C5)</c> — the caller
    /// owns and disposes all three.</summary>
    public (Tensor C3, Tensor C4, Tensor C5) Forward(IBackend backend, Tensor input)
    {
        Tensor s0 = _stem0.Forward(backend, input);
        Tensor s1 = _stem1.Forward(backend, s0); s0.Dispose();

        Tensor d3 = _down3.Forward(backend, s1); s1.Dispose();
        Tensor c3 = _c2f3.Forward(backend, d3); d3.Dispose();

        Tensor d4 = _down4.Forward(backend, c3);
        Tensor c4 = _c2f4.Forward(backend, d4); d4.Dispose();

        Tensor d5 = _down5.Forward(backend, c4);
        Tensor c5 = _c2f5.Forward(backend, d5); d5.Dispose();

        return (c3, c4, c5);
    }
}
