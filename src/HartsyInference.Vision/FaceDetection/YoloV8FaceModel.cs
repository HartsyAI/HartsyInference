using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection;
using HartsyInference.Vision.Detection.Blocks;

namespace HartsyInference.Vision.FaceDetection;

/// <summary>YOLOv8-Face model — the exact YOLOv8 backbone (Conv → C2f → SPPF) + FPN/PAN neck as
/// <see cref="YoloModel"/>, but with a <see cref="FaceDetectHead"/> (box + 1-class + 5-point landmark branch) in
/// place of the plain detect head. <see cref="Forward"/> returns both the box/class detections and the decoded
/// landmarks so a caller can NMS the boxes and gather each survivor's five points.
///
/// <para><b>Why a face-specific model instead of reusing the pose model.</b> The engine's existing keypoint head
/// (<see cref="Detection.YoloV11PoseModel"/> / <see cref="PoseHeadV11"/>) sits on the <i>v11</i> trunk (C3k2 +
/// C2PSA) and the v11 detect head. YOLOv8-Face checkpoints ship the <i>v8</i> trunk (C2f) and the v8 detect head,
/// with the landmark branch at layer 22 (v11 pose puts it at 23 because of the extra C2PSA layer). Rather than
/// distort the v11 pose path, this mirrors <see cref="YoloModel"/> (the tested v8 stack) and swaps in
/// <see cref="FaceDetectHead"/> — the landmark branch is the only new code, exactly as intended.</para>
///
/// <para>The backbone/neck body is duplicated from <see cref="YoloModel"/> because that class is <c>sealed</c> with a
/// private head; the shared block types (<see cref="C2f"/>, <see cref="Sppf"/>, <see cref="ConvBnSilu"/>) are reused
/// directly. This matches the existing precedent set by <see cref="Detection.YoloV11PoseModel"/>, which likewise
/// inlines the trunk rather than sharing it with the detector.</para></summary>
public sealed class YoloV8FaceModel
{
    private readonly YoloConfig _config;

    private readonly ConvBnSilu _layer0;
    private readonly ConvBnSilu _layer1;
    private readonly C2f _layer2;
    private readonly ConvBnSilu _layer3;
    private readonly C2f _layer4;
    private readonly ConvBnSilu _layer5;
    private readonly C2f _layer6;
    private readonly ConvBnSilu _layer7;
    private readonly C2f _layer8;
    private readonly Sppf _layer9;
    private readonly C2f _layer12;
    private readonly C2f _layer15;
    private readonly ConvBnSilu _layer16;
    private readonly C2f _layer18;
    private readonly ConvBnSilu _layer19;
    private readonly C2f _layer21;
    private readonly FaceDetectHead _head;

    /// <summary>The variant + scaling parameters this model was constructed with.</summary>
    public YoloConfig Config => _config;

    /// <summary>Detection classes (1 for a face detector).</summary>
    public int NumClasses => _config.NumClasses;

    /// <summary>Landmark points per face (5).</summary>
    public int NumKeypoints => _config.NumKeypoints;

    /// <summary>Values per landmark point (2 or 3).</summary>
    public int KptDims => _config.KptDims;

    /// <summary>Per-detect-scale strides — <c>[8, 16, 32]</c>.</summary>
    public IReadOnlyList<int> Strides => _config.Strides;

    public YoloV8FaceModel(YoloConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.NumKeypoints <= 0)
            throw new ArgumentException("YoloV8FaceModel requires a face config (NumKeypoints > 0).", nameof(config));
        _config = config;

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

        _layer12 = new C2f(p5 + p4, p4, rn0, shortcut: false);
        _layer15 = new C2f(p4 + p3, p3, rn1, shortcut: false);
        _layer16 = new ConvBnSilu(p3, strideH: 2, strideW: 2, padH: 1, padW: 1);
        _layer18 = new C2f(p3 + p4, p4, rn2, shortcut: false);
        _layer19 = new ConvBnSilu(p4, strideH: 2, strideW: 2, padH: 1, padW: 1);
        _layer21 = new C2f(p4 + p5, p5, rn3, shortcut: false);

        _head = new FaceDetectHead(
            numClasses: config.NumClasses,
            regMax: config.RegMax,
            numPoints: config.NumKeypoints,
            kptDims: config.KptDims,
            inChannels: [p3, p4, p5],
            strides: [config.Strides[0], config.Strides[1], config.Strides[2]]);
    }

    /// <summary>Loads BN-folded safetensors weights. Keys are Ultralytics <c>model.{layer}.*</c>; the face head lives
    /// at index 22 (v8 layout).</summary>
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
        _head.LoadWeights(weights, $"{prefix}.22");
    }

    /// <summary>Yields every weight tensor for GPU preload.</summary>
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
        foreach (Tensor t in _head.EnumerateWeights()) yield return t;
    }

    /// <summary>Backbone → neck → face head. Returns box/class detections <c>[1, 4+1, A]</c> and landmarks
    /// <c>[1, 5·ndim, A]</c> (both in letterbox-canvas pixels); caller owns both.</summary>
    public (Tensor detections, Tensor landmarks) Forward(IBackend backend, Tensor input)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(input);
        if (input.Shape.Rank != 4 || input.Shape[1] != 3)
            throw new ArgumentException($"YOLO input must be [N, 3, H, W]; got {input.Shape}.", nameof(input));

        Tensor x0 = _layer0.Forward(backend, input);
        Tensor x1 = _layer1.Forward(backend, x0); x0.Dispose();
        Tensor x2 = _layer2.Forward(backend, x1); x1.Dispose();
        Tensor x3 = _layer3.Forward(backend, x2); x2.Dispose();
        Tensor x4 = _layer4.Forward(backend, x3); x3.Dispose(); // P3 backbone
        Tensor x5 = _layer5.Forward(backend, x4);
        Tensor x6 = _layer6.Forward(backend, x5); x5.Dispose(); // P4 backbone
        Tensor x7 = _layer7.Forward(backend, x6);
        Tensor x8 = _layer8.Forward(backend, x7); x7.Dispose();
        Tensor x9 = _layer9.Forward(backend, x8); x8.Dispose(); // P5 SPPF

        Tensor x10 = Upsample(backend, x9, 2);
        Tensor x11 = ConcatChannel(backend, x10, x6);
        x10.Dispose();
        Tensor x12 = _layer12.Forward(backend, x11); x11.Dispose();

        Tensor x13 = Upsample(backend, x12, 2);
        Tensor x14 = ConcatChannel(backend, x13, x4);
        x13.Dispose(); x4.Dispose();
        Tensor x15 = _layer15.Forward(backend, x14); x14.Dispose(); // P3 detect input

        Tensor x16 = _layer16.Forward(backend, x15);
        Tensor x17 = ConcatChannel(backend, x16, x12);
        x16.Dispose(); x12.Dispose();
        Tensor x18 = _layer18.Forward(backend, x17); x17.Dispose(); // P4 detect input

        Tensor x19 = _layer19.Forward(backend, x18);
        Tensor x20 = ConcatChannel(backend, x19, x9);
        x19.Dispose(); x9.Dispose();
        Tensor x21 = _layer21.Forward(backend, x20); x20.Dispose(); // P5 detect input

        (Tensor detections, Tensor landmarks) = _head.Forward(backend, [x15, x18, x21]);
        x15.Dispose(); x18.Dispose(); x21.Dispose();
        return (detections, landmarks);
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
