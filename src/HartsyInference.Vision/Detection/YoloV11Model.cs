using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection.Blocks;

namespace HartsyInference.Vision.Detection;

/// <summary>YOLO11 detection model — 11-layer CSPDarknet backbone (Conv → C3k2 stages → SPPF →
/// C2PSA) + 12-layer FPN+PAN neck built with C3k2 blocks + decoupled detect head with DFL.
/// Layer indices follow the YOLO11 YAML so weight keys map 1:1 from the converted safetensors.
/// <para>Key architecture differences vs YOLOv8 (encoded in this class rather than the config so
/// the config record stays scaling-only):</para>
/// <list type="bullet">
///   <item>C3k2 replaces C2f. Early stages use <c>c3k=False, e=0.25</c> (narrow); deeper stages
///         (backbone 6, 8; neck 22) use <c>c3k=True, e=0.5</c>.</item>
///   <item>C2PSA (cross-stage position-sensitive attention) added at backbone layer 10, after
///         SPPF — provides spatial attention that captures longer-range dependencies than
///         pure convolutions.</item>
///   <item>Channel progression jumps a level early: layer 2's C3k2 outputs 256ch (vs C2f's 128ch
///         in v8), so the P3 detect head input matches base 256 (= 64 channels at n width).</item>
/// </list></summary>
public sealed class YoloV11Model : IYoloDetectModel
{
    private readonly YoloConfig _config;

    // Backbone (11 layers, indices 0..10). Internal so diagnostic tests can probe per-layer outputs.
    internal readonly ConvBnSilu _layer0;
    internal readonly ConvBnSilu _layer1;
    internal readonly C3k2 _layer2;
    internal readonly ConvBnSilu _layer3;
    internal readonly C3k2 _layer4;        // outputs feed neck concat 13 (P3 backbone)
    internal readonly ConvBnSilu _layer5;
    internal readonly C3k2 _layer6;        // outputs feed neck concat 11 (P4 backbone)
    internal readonly ConvBnSilu _layer7;
    internal readonly C3k2 _layer8;
    internal readonly Sppf _layer9;
    internal readonly C2psa _layer10;      // outputs feed neck concat 20 (P5 SPPF+C2PSA)

    // Neck — internal so the layer-by-layer diagnostic test can probe each output.
    internal readonly C3k2 _layer13;       // C3k2 after upsample(layer10) + concat(layer6)
    internal readonly C3k2 _layer16;       // C3k2 after upsample(layer13) + concat(layer4) → P3 detect input
    internal readonly ConvBnSilu _layer17; // downsample
    internal readonly C3k2 _layer19;       // → P4 detect input
    internal readonly ConvBnSilu _layer20; // downsample
    internal readonly C3k2 _layer22;       // → P5 detect input (uses c3k=True)

    // Detect head (layer 23 in YOLO11 YAML).
    private readonly DetectHeadV11 _detect;

    public YoloConfig Config => _config;
    public int NumClasses => _config.NumClasses;
    public IReadOnlyList<int> Strides => _config.Strides;

    /// <summary>Constructs YOLO11 model from a YoloConfig (use <see cref="YoloConfig.YoloV11n"/>
    /// or similar preset). Channel counts are pre-resolved here.</summary>
    public YoloV11Model(YoloConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;

        // Resolved channel counts at each backbone "level".
        // YOLO11 YAML output channels per layer:
        //   0: 64,  1: 128, 2: 256, 3: 256, 4: 512, 5: 512, 6: 512, 7: 1024, 8: 1024, 9: 1024, 10: 1024
        int c64 = ScaleCh(config, 64);
        int c128 = ScaleCh(config, 128);
        int c256 = ScaleCh(config, 256);
        int c512 = ScaleCh(config, 512);
        int c1024 = ScaleCh(config, 1024);

        // C3k2 repeat counts (n=2 base for all C3k2 blocks).
        int r2 = Math.Max((int)MathF.Round(2 * config.DepthMultiple), 1);

        _layer0 = new ConvBnSilu(c64, 2, 2, 1, 1);
        _layer1 = new ConvBnSilu(c128, 2, 2, 1, 1);
        _layer2 = new C3k2(c128, c256, r2, c3k: false, shortcut: true, expansion: 0.25f);
        _layer3 = new ConvBnSilu(c256, 2, 2, 1, 1);
        _layer4 = new C3k2(c256, c512, r2, c3k: false, shortcut: true, expansion: 0.25f);
        _layer5 = new ConvBnSilu(c512, 2, 2, 1, 1);
        _layer6 = new C3k2(c512, c512, r2, c3k: true, shortcut: true);
        _layer7 = new ConvBnSilu(c1024, 2, 2, 1, 1);
        _layer8 = new C3k2(c1024, c1024, r2, c3k: true, shortcut: true);
        _layer9 = new Sppf(c1024, c1024, kernel: 5);
        _layer10 = new C2psa(c1024, r2);

        // Neck. Layer 13 in YAML has output 512 base, layer 16 has 256, layer 19 has 512, layer 22 has 1024.
        // Input channel widths after concat:
        //   13 input = upsample(layer10 = c1024) + layer6 (c512) = c1024 + c512
        //   16 input = upsample(layer13 = c512)  + layer4 (c512) = c512 + c512
        //   19 input = downsample(layer16 = c256) + layer13 (c512) = c256 + c512
        //   22 input = downsample(layer19 = c512) + layer10 (c1024) = c512 + c1024
        _layer13 = new C3k2(c1024 + c512, c512, r2, c3k: false, shortcut: true);
        _layer16 = new C3k2(c512 + c512, c256, r2, c3k: false, shortcut: true);
        _layer17 = new ConvBnSilu(c256, 2, 2, 1, 1);
        _layer19 = new C3k2(c256 + c512, c512, r2, c3k: false, shortcut: true);
        _layer20 = new ConvBnSilu(c512, 2, 2, 1, 1);
        _layer22 = new C3k2(c512 + c1024, c1024, r2, c3k: true, shortcut: true);

        _detect = new DetectHeadV11(
            numClasses: config.NumClasses,
            regMax: config.RegMax,
            inChannels: [c256, c512, c1024],
            strides: [config.Strides[0], config.Strides[1], config.Strides[2]]);
    }

    /// <summary>Applies Ultralytics scaling for a YAML-listed base channel count.</summary>
    private static int ScaleCh(YoloConfig config, int baseCh)
    {
        int capped = Math.Min(baseCh, config.MaxChannels);
        int scaled = (int)MathF.Round(capped * config.WidthMultiple);
        // make_divisible(x, 8) — ceiling-round to multiple of 8.
        return ((scaled + 7) / 8) * 8;
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
        _layer10.LoadWeights(weights, $"{prefix}.10");
        // Layers 11, 12 = Upsample + Concat.
        _layer13.LoadWeights(weights, $"{prefix}.13");
        // Layers 14, 15 = Upsample + Concat.
        _layer16.LoadWeights(weights, $"{prefix}.16");
        _layer17.LoadWeights(weights, $"{prefix}.17.conv");
        // Layer 18 = Concat.
        _layer19.LoadWeights(weights, $"{prefix}.19");
        _layer20.LoadWeights(weights, $"{prefix}.20.conv");
        // Layer 21 = Concat.
        _layer22.LoadWeights(weights, $"{prefix}.22");
        _detect.LoadWeights(weights, $"{prefix}.23");
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
        foreach (Tensor t in _layer10.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer13.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer16.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer17.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer19.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer20.EnumerateWeights()) yield return t;
        foreach (Tensor t in _layer22.EnumerateWeights()) yield return t;
        foreach (Tensor t in _detect.EnumerateWeights()) yield return t;
    }

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
        Tensor x4 = _layer4.Forward(backend, x3); x3.Dispose(); // P3 backbone — feeds layer 15 concat
        Tensor x5 = _layer5.Forward(backend, x4);
        Tensor x6 = _layer6.Forward(backend, x5); x5.Dispose(); // P4 backbone — feeds layer 12 concat
        Tensor x7 = _layer7.Forward(backend, x6);
        Tensor x8 = _layer8.Forward(backend, x7); x7.Dispose();
        Tensor x9 = _layer9.Forward(backend, x8); x8.Dispose();
        Tensor x10 = _layer10.Forward(backend, x9); x9.Dispose(); // P5 SPPF+C2PSA — feeds layer 21 concat

        // FPN top-down.
        Tensor x11 = Upsample(backend, x10, 2);
        Tensor x12 = ConcatChannel(backend, x11, x6); x11.Dispose();
        Tensor x13 = _layer13.Forward(backend, x12); x12.Dispose();

        Tensor x14 = Upsample(backend, x13, 2);
        Tensor x15 = ConcatChannel(backend, x14, x4); x14.Dispose(); x4.Dispose();
        Tensor x16 = _layer16.Forward(backend, x15); x15.Dispose(); // P3 detect input

        // PAN bottom-up.
        Tensor x17 = _layer17.Forward(backend, x16);
        Tensor x18 = ConcatChannel(backend, x17, x13); x17.Dispose(); x13.Dispose();
        Tensor x19 = _layer19.Forward(backend, x18); x18.Dispose(); // P4 detect input

        Tensor x20 = _layer20.Forward(backend, x19);
        Tensor x21 = ConcatChannel(backend, x20, x10); x20.Dispose(); x10.Dispose();
        Tensor x22 = _layer22.Forward(backend, x21); x21.Dispose(); // P5 detect input

        Tensor decoded = _detect.Forward(backend, [x16, x19, x22]);
        x16.Dispose(); x19.Dispose(); x22.Dispose();
        return decoded;
    }

    internal static Tensor Upsample(IBackend backend, Tensor input, int scale)
    {
        int n = (int)input.Shape[0];
        int c = (int)input.Shape[1];
        int h = (int)input.Shape[2];
        int w = (int)input.Shape[3];
        Tensor output = new Tensor(new TensorShape(n, c, h * scale, w * scale), DType.F32);
        backend.UpsampleNearest2D(output, input, scale, scale);
        return output;
    }

    internal static Tensor ConcatChannel(IBackend backend, Tensor a, Tensor b)
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
