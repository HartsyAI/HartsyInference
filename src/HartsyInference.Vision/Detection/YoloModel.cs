using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection.Blocks;

namespace HartsyInference.Vision.Detection;

/// <summary>End-to-end YOLOv8 model: 10-layer CSPDarknet-style backbone (Conv → C2f stages →
/// SPPF) + 12-layer FPN+PAN neck + decoupled detect head with DFL decoding. Layer indices follow
/// the Ultralytics YAML so weight names map 1:1 (<c>model.0.conv.weight</c>,
/// <c>model.2.cv1.conv.weight</c>, …).
/// <para>Input is <c>[1, 3, H, W]</c> where H and W are multiples of 32 (the network's max stride).
/// Output is <c>[1, 4 + numClasses, totalAnchors]</c> with box xywh in input-pixel coords and
/// class probabilities after sigmoid — ready for confidence filtering and NMS.</para></summary>
public sealed class YoloModel : IYoloDetectModel
{
    private readonly YoloConfig _config;

    // Backbone (10 layers, indices 0..9 in Ultralytics YAML).
    private readonly ConvBnSilu _layer0;   // 0: Conv (3 → P1, k=3, s=2)
    private readonly ConvBnSilu _layer1;   // 1: Conv (P1 → P2, k=3, s=2)
    private readonly C2f _layer2;          // 2: C2f at P2/4
    private readonly ConvBnSilu _layer3;   // 3: Conv (P2 → P3, k=3, s=2)
    private readonly C2f _layer4;          // 4: C2f at P3/8 (output feeds neck concat 14)
    private readonly ConvBnSilu _layer5;   // 5: Conv (P3 → P4, k=3, s=2)
    private readonly C2f _layer6;          // 6: C2f at P4/16 (output feeds neck concat 11)
    private readonly ConvBnSilu _layer7;   // 7: Conv (P4 → P5, k=3, s=2)
    private readonly C2f _layer8;          // 8: C2f at P5/32
    private readonly Sppf _layer9;         // 9: SPPF (output feeds neck concat 20)

    // Neck — layer indices match the YAML; non-weighted ops (Upsample, Concat) are inline.
    private readonly C2f _layer12;         // 12: C2f after upsample(layer9) + concat(layer6)
    private readonly C2f _layer15;         // 15: C2f after upsample(layer12) + concat(layer4) → P3 detect input
    private readonly ConvBnSilu _layer16;  // 16: Conv downsample of layer15
    private readonly C2f _layer18;         // 18: C2f after concat(layer16, layer12) → P4 detect input
    private readonly ConvBnSilu _layer19;  // 19: Conv downsample of layer18
    private readonly C2f _layer21;         // 21: C2f after concat(layer19, layer9) → P5 detect input

    // Detect head (layer 22).
    private readonly DetectHead _detect;

    /// <summary>The variant + scaling parameters this model was constructed with.</summary>
    public YoloConfig Config => _config;

    /// <summary>Number of classes the head was built for.</summary>
    public int NumClasses => _config.NumClasses;

    /// <summary>Per-detect-scale strides — <c>[8, 16, 32]</c> for YOLOv8.</summary>
    public IReadOnlyList<int> Strides => _config.Strides;

    /// <summary>Constructs the model from a config. Channel counts are pre-resolved here so
    /// downstream callers don't need to apply width/depth multipliers themselves.</summary>
    public YoloModel(YoloConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;

        // Backbone stage channels (after width scaling). C2f bottleneck repeats after depth scaling.
        int p1 = config.ScaledChannel(0);
        int p2 = config.ScaledChannel(1);
        int p3 = config.ScaledChannel(2);
        int p4 = config.ScaledChannel(3);
        int p5 = config.ScaledChannel(4);

        int r0 = config.ScaledBackboneRepeat(0); // C2f layer 2
        int r1 = config.ScaledBackboneRepeat(1); // C2f layer 4
        int r2 = config.ScaledBackboneRepeat(2); // C2f layer 6
        int r3 = config.ScaledBackboneRepeat(3); // C2f layer 8

        int rn0 = config.ScaledNeckRepeat(0); // C2f layer 12
        int rn1 = config.ScaledNeckRepeat(1); // C2f layer 15
        int rn2 = config.ScaledNeckRepeat(2); // C2f layer 18
        int rn3 = config.ScaledNeckRepeat(3); // C2f layer 21

        _layer0 = new ConvBnSilu(p1, strideH: 2, strideW: 2, padH: 1, padW: 1);
        _layer1 = new ConvBnSilu(p2, strideH: 2, strideW: 2, padH: 1, padW: 1);
        _layer2 = new C2f(p2, p2, r0, shortcut: true);
        _layer3 = new ConvBnSilu(p3, strideH: 2, strideW: 2, padH: 1, padW: 1);
        _layer4 = new C2f(p3, p3, r1, shortcut: true);
        _layer5 = new ConvBnSilu(p4, strideH: 2, strideW: 2, padH: 1, padW: 1);
        _layer6 = new C2f(p4, p4, r2, shortcut: true);
        _layer7 = new ConvBnSilu(p5, strideH: 2, strideW: 2, padH: 1, padW: 1);
        _layer8 = new C2f(p5, p5, r3, shortcut: true);
        _layer9 = new Sppf(p5, p5, kernel: 5);

        // Neck — channels follow Ultralytics conventions:
        //   layer 12 in = (p5 from layer9 upsampled) + (p4 from layer6 backbone) = p5 + p4 concatenated
        //   layer 12 out = p4
        //   layer 15 in = (p4 from layer12 upsampled) + (p3 from layer4 backbone) = p4 + p3
        //   layer 15 out = p3
        //   layer 18 in = (p3 from layer16 downsampled) + (p4 from layer12) = p3 + p4
        //   layer 18 out = p4
        //   layer 21 in = (p4 from layer19 downsampled) + (p5 from layer9 SPPF) = p4 + p5
        //   layer 21 out = p5
        _layer12 = new C2f(p5 + p4, p4, rn0, shortcut: false);
        _layer15 = new C2f(p4 + p3, p3, rn1, shortcut: false);
        _layer16 = new ConvBnSilu(p3, strideH: 2, strideW: 2, padH: 1, padW: 1);
        _layer18 = new C2f(p3 + p4, p4, rn2, shortcut: false);
        _layer19 = new ConvBnSilu(p4, strideH: 2, strideW: 2, padH: 1, padW: 1);
        _layer21 = new C2f(p4 + p5, p5, rn3, shortcut: false);

        _detect = new DetectHead(
            numClasses: config.NumClasses,
            regMax: config.RegMax,
            inChannels: [p3, p4, p5],
            strides: [config.Strides[0], config.Strides[1], config.Strides[2]]);
    }

    /// <summary>Loads weights from an already-BN-folded safetensors dict. Keys are
    /// Ultralytics-format <c>model.{layer_index}.*</c>. The detect head lives at index 22.</summary>
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

        // Layers 10, 11 = Upsample + Concat (no weights). Layer 12 = C2f.
        _layer12.LoadWeights(weights, $"{prefix}.12");
        // Layers 13, 14 = Upsample + Concat (no weights). Layer 15 = C2f.
        _layer15.LoadWeights(weights, $"{prefix}.15");
        _layer16.LoadWeights(weights, $"{prefix}.16.conv");
        // Layer 17 = Concat. Layer 18 = C2f.
        _layer18.LoadWeights(weights, $"{prefix}.18");
        _layer19.LoadWeights(weights, $"{prefix}.19.conv");
        // Layer 20 = Concat. Layer 21 = C2f.
        _layer21.LoadWeights(weights, $"{prefix}.21");

        _detect.LoadWeights(weights, $"{prefix}.22");
    }

    /// <summary>Yields every weight tensor for GPU preloading.</summary>
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
        foreach (Tensor t in _detect.EnumerateWeights()) yield return t;
    }

    /// <summary>Runs the full forward pass: backbone → FPN+PAN neck → detect head + DFL.
    /// Caller owns the returned tensor.</summary>
    public Tensor Forward(IBackend backend, Tensor input)
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
        Tensor x4 = _layer4.Forward(backend, x3); x3.Dispose(); // P3 backbone — feeds neck concat 14
        Tensor x5 = _layer5.Forward(backend, x4);
        Tensor x6 = _layer6.Forward(backend, x5); x5.Dispose(); // P4 backbone — feeds neck concat 11
        Tensor x7 = _layer7.Forward(backend, x6);
        Tensor x8 = _layer8.Forward(backend, x7); x7.Dispose();
        Tensor x9 = _layer9.Forward(backend, x8); x8.Dispose(); // P5 SPPF — feeds neck concat 20

        // Neck — FPN top-down path.
        Tensor x10 = YoloV11Model.Upsample(backend, x9, scale: 2);
        Tensor x11 = YoloV11Model.ConcatChannel(backend, x10, x6);
        x10.Dispose(); x6.Dispose();
        Tensor x12 = _layer12.Forward(backend, x11); x11.Dispose();

        Tensor x13 = YoloV11Model.Upsample(backend, x12, scale: 2);
        Tensor x14 = YoloV11Model.ConcatChannel(backend, x13, x4);
        x13.Dispose(); x4.Dispose();
        Tensor x15 = _layer15.Forward(backend, x14); x14.Dispose(); // P3 detect input

        // Neck — PAN bottom-up path.
        Tensor x16 = _layer16.Forward(backend, x15);
        Tensor x17 = YoloV11Model.ConcatChannel(backend, x16, x12);
        x16.Dispose(); x12.Dispose();
        Tensor x18 = _layer18.Forward(backend, x17); x17.Dispose(); // P4 detect input

        Tensor x19 = _layer19.Forward(backend, x18);
        Tensor x20 = YoloV11Model.ConcatChannel(backend, x19, x9);
        x19.Dispose(); x9.Dispose();
        Tensor x21 = _layer21.Forward(backend, x20); x20.Dispose(); // P5 detect input

        // Detect head expects [P3, P4, P5] in that order.
        Tensor decoded = _detect.Forward(backend, [x15, x18, x21]);
        x15.Dispose(); x18.Dispose(); x21.Dispose();
        return decoded;
    }

}
