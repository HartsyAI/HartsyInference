using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection.Blocks;

namespace HartsyInference.Vision.Detection;

/// <summary>Multi-scale image backbone for Grounding DINO — a lightweight CNN <b>structural stand-in</b> for the real
/// Swin-T hierarchical transformer. Five stride-2 stages (total stride 32); the last three (strides 8 / 16 / 32) are
/// the feature levels, each refined by a <see cref="C2f"/> and projected to the shared hidden width by a 1×1 conv.
///
/// <para>The point of this phase is the cross-modality feature enhancer and the open-vocabulary head, not backbone
/// fidelity: a real Grounding DINO would replace this with Swin-T and keep everything downstream. Because the
/// forward reuses <see cref="ConvBnSilu"/>/<see cref="C2f"/>, whose weight schema is fixed by
/// <c>Blocks/C2f.cs</c>, the synthetic/real weight schema below mirrors that schedule (hidden = out·0.5,
/// concat = (2+n)·hidden). If C2f's internal channel math changes, this mirror must track it.</para></summary>
public sealed class GroundingDinoBackbone
{
    private const int BackboneC2fRepeats = 1;

    private readonly GroundingDinoConfig _config;
    private readonly int _hidden;
    private readonly int[] _ch;

    private readonly ConvBnSilu _d0, _d1, _d2, _d3, _d4;
    private readonly C2f _c2f3, _c2f4, _c2f5;
    private readonly ConvBnSilu _proj3, _proj4, _proj5;

    /// <summary>Number of feature levels emitted (3).</summary>
    public int NumLevels => 3;

    public GroundingDinoBackbone(GroundingDinoConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.BackboneChannels.Length != 5)
            throw new ArgumentException("BackboneChannels must have 5 entries (five stride-2 stages).", nameof(config));
        _config = config;
        _hidden = config.ImageHiddenDim;
        _ch = config.BackboneChannels;

        _d0 = new ConvBnSilu(_ch[0], strideH: 2, strideW: 2, padH: 1, padW: 1);
        _d1 = new ConvBnSilu(_ch[1], strideH: 2, strideW: 2, padH: 1, padW: 1);
        _d2 = new ConvBnSilu(_ch[2], strideH: 2, strideW: 2, padH: 1, padW: 1);
        _c2f3 = new C2f(_ch[2], _ch[2], BackboneC2fRepeats, shortcut: true);
        _d3 = new ConvBnSilu(_ch[3], strideH: 2, strideW: 2, padH: 1, padW: 1);
        _c2f4 = new C2f(_ch[3], _ch[3], BackboneC2fRepeats, shortcut: true);
        _d4 = new ConvBnSilu(_ch[4], strideH: 2, strideW: 2, padH: 1, padW: 1);
        _c2f5 = new C2f(_ch[4], _ch[4], BackboneC2fRepeats, shortcut: true);

        // 1×1 channel projections to the shared hidden width; no activation (they feed the transformer directly).
        _proj3 = new ConvBnSilu(_hidden, strideH: 1, strideW: 1, padH: 0, padW: 0, useSilu: false);
        _proj4 = new ConvBnSilu(_hidden, strideH: 1, strideW: 1, padH: 0, padW: 0, useSilu: false);
        _proj5 = new ConvBnSilu(_hidden, strideH: 1, strideW: 1, padH: 0, padW: 0, useSilu: false);
    }

    /// <summary>Resolves every conv weight through the bag, then feeds the (block-keyed) tensors to the reused
    /// <see cref="ConvBnSilu"/>/<see cref="C2f"/> loaders.</summary>
    public void Build(IGroundingWeightBag bag, string prefix)
    {
        ArgumentNullException.ThrowIfNull(bag);
        Dictionary<string, Tensor> dict = new();

        RegisterConv(bag, dict, $"{prefix}.d0.conv", _ch[0], 3, 3);
        RegisterConv(bag, dict, $"{prefix}.d1.conv", _ch[1], _ch[0], 3);
        RegisterConv(bag, dict, $"{prefix}.d2.conv", _ch[2], _ch[1], 3);
        RegisterC2f(bag, dict, $"{prefix}.c2f3", _ch[2], _ch[2], BackboneC2fRepeats);
        RegisterConv(bag, dict, $"{prefix}.d3.conv", _ch[3], _ch[2], 3);
        RegisterC2f(bag, dict, $"{prefix}.c2f4", _ch[3], _ch[3], BackboneC2fRepeats);
        RegisterConv(bag, dict, $"{prefix}.d4.conv", _ch[4], _ch[3], 3);
        RegisterC2f(bag, dict, $"{prefix}.c2f5", _ch[4], _ch[4], BackboneC2fRepeats);
        RegisterConv(bag, dict, $"{prefix}.proj3.conv", _hidden, _ch[2], 1);
        RegisterConv(bag, dict, $"{prefix}.proj4.conv", _hidden, _ch[3], 1);
        RegisterConv(bag, dict, $"{prefix}.proj5.conv", _hidden, _ch[4], 1);

        _d0.LoadWeights(dict, $"{prefix}.d0.conv");
        _d1.LoadWeights(dict, $"{prefix}.d1.conv");
        _d2.LoadWeights(dict, $"{prefix}.d2.conv");
        _c2f3.LoadWeights(dict, $"{prefix}.c2f3");
        _d3.LoadWeights(dict, $"{prefix}.d3.conv");
        _c2f4.LoadWeights(dict, $"{prefix}.c2f4");
        _d4.LoadWeights(dict, $"{prefix}.d4.conv");
        _c2f5.LoadWeights(dict, $"{prefix}.c2f5");
        _proj3.LoadWeights(dict, $"{prefix}.proj3.conv");
        _proj4.LoadWeights(dict, $"{prefix}.proj4.conv");
        _proj5.LoadWeights(dict, $"{prefix}.proj5.conv");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _d0.EnumerateWeights()) yield return t;
        foreach (Tensor t in _d1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _d2.EnumerateWeights()) yield return t;
        foreach (Tensor t in _c2f3.EnumerateWeights()) yield return t;
        foreach (Tensor t in _d3.EnumerateWeights()) yield return t;
        foreach (Tensor t in _c2f4.EnumerateWeights()) yield return t;
        foreach (Tensor t in _d4.EnumerateWeights()) yield return t;
        foreach (Tensor t in _c2f5.EnumerateWeights()) yield return t;
        foreach (Tensor t in _proj3.EnumerateWeights()) yield return t;
        foreach (Tensor t in _proj4.EnumerateWeights()) yield return t;
        foreach (Tensor t in _proj5.EnumerateWeights()) yield return t;
    }

    /// <summary>Runs the backbone on <paramref name="pixelValues"/> <c>[1, 3, H, W]</c>; returns three
    /// hidden-width feature maps <c>[1, hidden, H/8, W/8]</c>, <c>[.../16]</c>, <c>[.../32]</c>. Caller owns them.</summary>
    public Tensor[] Forward(IBackend backend, Tensor pixelValues)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (pixelValues.Shape.Rank != 4 || pixelValues.Shape[1] != 3)
            throw new ArgumentException($"Backbone input must be [1, 3, H, W]; got {pixelValues.Shape}.", nameof(pixelValues));

        Tensor x0 = _d0.Forward(backend, pixelValues);
        Tensor x1 = _d1.Forward(backend, x0); x0.Dispose();
        Tensor x2 = _d2.Forward(backend, x1); x1.Dispose();

        Tensor p3 = _c2f3.Forward(backend, x2); x2.Dispose();
        Tensor x3 = _d3.Forward(backend, p3);
        Tensor p4 = _c2f4.Forward(backend, x3); x3.Dispose();
        Tensor x4 = _d4.Forward(backend, p4);
        Tensor p5 = _c2f5.Forward(backend, x4); x4.Dispose();

        Tensor f3 = _proj3.Forward(backend, p3); p3.Dispose();
        Tensor f4 = _proj4.Forward(backend, p4); p4.Dispose();
        Tensor f5 = _proj5.Forward(backend, p5); p5.Dispose();
        return [f3, f4, f5];
    }

    private static void RegisterConv(IGroundingWeightBag bag, Dictionary<string, Tensor> dict, string convPrefix, int cout, int cin, int k)
    {
        dict[$"{convPrefix}.weight"] = bag.Get($"{convPrefix}.weight", cout, cin, k, k);
        dict[$"{convPrefix}.bias"] = bag.Get($"{convPrefix}.bias", cout);
    }

    private static void RegisterC2f(IGroundingWeightBag bag, Dictionary<string, Tensor> dict, string prefix, int inC, int outC, int n)
    {
        int hidden = (int)(outC * 0.5f);
        RegisterConv(bag, dict, $"{prefix}.cv1.conv", 2 * hidden, inC, 1);
        RegisterConv(bag, dict, $"{prefix}.cv2.conv", outC, (2 + n) * hidden, 1);
        for (int i = 0; i < n; i++)
        {
            RegisterConv(bag, dict, $"{prefix}.m.{i}.cv1.conv", hidden, hidden, 3);
            RegisterConv(bag, dict, $"{prefix}.m.{i}.cv2.conv", hidden, hidden, 3);
        }
    }
}
