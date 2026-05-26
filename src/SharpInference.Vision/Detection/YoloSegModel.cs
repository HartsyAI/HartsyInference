using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Vision.Detection.Blocks;

namespace SharpInference.Vision.Detection;

/// <summary>YOLOv8-seg instance segmentation model — same 10-layer backbone + 12-layer FPN+PAN
/// neck as <see cref="YoloModel"/>, but the detect head is replaced by <see cref="SegmentDetectHead"/>
/// which adds a cv4 mask-coefficient branch and a Proto module fed from P3.
/// <para>Returns <c>(detections, prototypes)</c> instead of a single tensor. The per-detection
/// masks are assembled by the pipeline via <see cref="MaskAssembly.AssembleMask"/> after NMS.</para>
/// <para>This duplicates most of <see cref="YoloModel"/>'s layer construction. The backbone +
/// neck layers are identical; only the final head differs. A future refactor can extract a shared
/// <c>YoloBackboneNeck</c> base, but for clarity right now the two model classes live side by
/// side with copied wiring.</para></summary>
public sealed class YoloSegModel : IYoloDetectModel
{
    private readonly YoloConfig _config;
    private readonly int _numMasks;

    // Backbone (10 layers, indices 0..9 in Ultralytics YAML).
    private readonly ConvBnSilu _layer0, _layer1, _layer3, _layer5, _layer7;
    private readonly C2f _layer2, _layer4, _layer6, _layer8;
    private readonly Sppf _layer9;
    // Neck.
    private readonly C2f _layer12, _layer15, _layer18, _layer21;
    private readonly ConvBnSilu _layer16, _layer19;
    // Segmentation head (layer 22 in YAML).
    private readonly SegmentDetectHead _segHead;

    public YoloConfig Config => _config;
    public int NumClasses => _config.NumClasses;
    public int NumMasks => _numMasks;
    public IReadOnlyList<int> Strides => _config.Strides;
    public SegmentDetectHead Head => _segHead;

    public YoloSegModel(YoloConfig config, int numMasks = 32)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _numMasks = numMasks;

        int p1 = config.ScaledChannel(0);
        int p2 = config.ScaledChannel(1);
        int p3 = config.ScaledChannel(2);
        int p4 = config.ScaledChannel(3);
        int p5 = config.ScaledChannel(4);

        int r0 = config.ScaledBackboneRepeat(0);
        int r1 = config.ScaledBackboneRepeat(1);
        int r2 = config.ScaledBackboneRepeat(2);
        int r3 = config.ScaledBackboneRepeat(3);
        int rn0 = config.ScaledNeckRepeat(0);
        int rn1 = config.ScaledNeckRepeat(1);
        int rn2 = config.ScaledNeckRepeat(2);
        int rn3 = config.ScaledNeckRepeat(3);

        _layer0 = new ConvBnSilu(p1, 2, 2, 1, 1);
        _layer1 = new ConvBnSilu(p2, 2, 2, 1, 1);
        _layer2 = new C2f(p2, p2, r0, shortcut: true);
        _layer3 = new ConvBnSilu(p3, 2, 2, 1, 1);
        _layer4 = new C2f(p3, p3, r1, shortcut: true);
        _layer5 = new ConvBnSilu(p4, 2, 2, 1, 1);
        _layer6 = new C2f(p4, p4, r2, shortcut: true);
        _layer7 = new ConvBnSilu(p5, 2, 2, 1, 1);
        _layer8 = new C2f(p5, p5, r3, shortcut: true);
        _layer9 = new Sppf(p5, p5, kernel: 5);

        _layer12 = new C2f(p5 + p4, p4, rn0, shortcut: false);
        _layer15 = new C2f(p4 + p3, p3, rn1, shortcut: false);
        _layer16 = new ConvBnSilu(p3, 2, 2, 1, 1);
        _layer18 = new C2f(p3 + p4, p4, rn2, shortcut: false);
        _layer19 = new ConvBnSilu(p4, 2, 2, 1, 1);
        _layer21 = new C2f(p4 + p5, p5, rn3, shortcut: false);

        // Proto internal channels = ch[0] (per Ultralytics' default npr=256 scaled by width to ch[0]
        // for the standard variants — for v8n with ch[0]=64, npr=64. The seg checkpoint confirms
        // the proto.cv1 weight is [64, 64, 3, 3], so npr=64).
        int protoNpr = p3;
        _segHead = new SegmentDetectHead(
            numClasses: config.NumClasses,
            numMasks: _numMasks,
            regMax: config.RegMax,
            inChannels: [p3, p4, p5],
            strides: [config.Strides[0], config.Strides[1], config.Strides[2]],
            protoInternalChannels: protoNpr);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "model")
    {
        _layer0.LoadWeights(weights, $"{prefix}.0.conv");
        _layer1.LoadWeights(weights, $"{prefix}.1.conv");
        _layer2.LoadWeights(weights, $"{prefix}.2");
        _layer3.LoadWeights(weights, $"{prefix}.3.conv");
        _layer4.LoadWeights(weights, $"{prefix}.4");
        _layer5.LoadWeights(weights, $"{prefix}.5.conv");
        _layer6.LoadWeights(weights, $"{prefix}.6");
        _layer7.LoadWeights(weights, $"{prefix}.7.conv");
        _layer8.LoadWeights(weights, $"{prefix}.8");
        _layer9.LoadWeights(weights, $"{prefix}.9");
        _layer12.LoadWeights(weights, $"{prefix}.12");
        _layer15.LoadWeights(weights, $"{prefix}.15");
        _layer16.LoadWeights(weights, $"{prefix}.16.conv");
        _layer18.LoadWeights(weights, $"{prefix}.18");
        _layer19.LoadWeights(weights, $"{prefix}.19.conv");
        _layer21.LoadWeights(weights, $"{prefix}.21");
        _segHead.LoadWeights(weights, $"{prefix}.22");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _layer0.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer2.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer3.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer4.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer5.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer6.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer7.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer8.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer9.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer12.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer15.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer16.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer18.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer19.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer21.EnumerateWeights()) yield return t;
        foreach (Tensor t in _segHead.EnumerateWeights()) yield return t;
    }

    /// <summary><see cref="IYoloDetectModel.Forward"/> — returns the box+class+mask-coef tensor only.
    /// Use <see cref="ForwardWithProtos"/> when you need both the detections and the mask prototypes.</summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        (Tensor detections, Tensor protos) = ForwardWithProtos(backend, input);
        protos.Dispose();
        return detections;
    }

    /// <summary>Runs the full seg model. Returns both the detections tensor
    /// <c>[B, 4 + nc + nm, totalAnchors]</c> and the mask prototypes <c>[B, nm, protoH, protoW]</c>.</summary>
    public (Tensor detections, Tensor prototypes) ForwardWithProtos(IBackend backend, Tensor input)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(input);
        if (input.Shape.Rank != 4 || input.Shape[1] != 3)
            throw new ArgumentException($"YOLO input must be [N, 3, H, W]; got {input.Shape}.", nameof(input));

        // Backbone.
        Tensor x0 = _layer0.Forward(backend, input);
        Tensor x1 = _layer1.Forward(backend, x0); x0.Dispose();
        Tensor x2 = _layer2.Forward(backend, x1); x1.Dispose();
        Tensor x3 = _layer3.Forward(backend, x2); x2.Dispose();
        Tensor x4 = _layer4.Forward(backend, x3); x3.Dispose();
        Tensor x5 = _layer5.Forward(backend, x4);
        Tensor x6 = _layer6.Forward(backend, x5); x5.Dispose();
        Tensor x7 = _layer7.Forward(backend, x6);
        Tensor x8 = _layer8.Forward(backend, x7); x7.Dispose();
        Tensor x9 = _layer9.Forward(backend, x8); x8.Dispose();

        // Neck.
        Tensor x10 = Upsample(backend, x9, 2);
        Tensor x11 = ConcatChannel(backend, x10, x6); x10.Dispose();
        Tensor x12 = _layer12.Forward(backend, x11); x11.Dispose();

        Tensor x13 = Upsample(backend, x12, 2);
        Tensor x14 = ConcatChannel(backend, x13, x4); x13.Dispose(); x4.Dispose();
        Tensor x15 = _layer15.Forward(backend, x14); x14.Dispose();

        Tensor x16 = _layer16.Forward(backend, x15);
        Tensor x17 = ConcatChannel(backend, x16, x12); x16.Dispose(); x12.Dispose();
        Tensor x18 = _layer18.Forward(backend, x17); x17.Dispose();

        Tensor x19 = _layer19.Forward(backend, x18);
        Tensor x20 = ConcatChannel(backend, x19, x9); x19.Dispose(); x9.Dispose();
        Tensor x21 = _layer21.Forward(backend, x20); x20.Dispose();

        (Tensor detections, Tensor protos) = _segHead.Forward(backend, [x15, x18, x21]);
        x15.Dispose(); x18.Dispose(); x21.Dispose();
        return (detections, protos);
    }

    private static Tensor Upsample(IBackend backend, Tensor input, int scale)
    {
        int n = (int)input.Shape[0];
        int c = (int)input.Shape[1];
        int h = (int)input.Shape[2];
        int w = (int)input.Shape[3];
        Tensor output = new Tensor(new TensorShape(n, c, h * scale, w * scale), DType.F32);
        backend.UpsampleNearest2D(output, input, scale, scale);
        return output;
    }

    private static Tensor ConcatChannel(IBackend backend, Tensor a, Tensor b)
    {
        int n = (int)a.Shape[0];
        int c = (int)(a.Shape[1] + b.Shape[1]);
        int h = (int)a.Shape[2];
        int w = (int)a.Shape[3];
        Tensor output = new Tensor(new TensorShape(n, c, h, w), DType.F32);
        backend.Concat(output, [a, b], dim: 1);
        return output;
    }
}
