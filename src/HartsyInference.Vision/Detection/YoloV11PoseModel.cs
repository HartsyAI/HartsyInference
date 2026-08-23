using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection.Blocks;

namespace HartsyInference.Vision.Detection;

/// <summary>YOLO11-pose model — the same CSPDarknet backbone + FPN/PAN neck as <see cref="YoloV11Model"/> with a
/// <see cref="PoseHeadV11"/> (box + class + 17-keypoint branches). <see cref="Forward"/> returns both the box/class
/// detections and the decoded keypoints so a caller can NMS the boxes and gather each survivor's keypoints.
///
/// <para>TODO(refactor): the backbone + neck are shared with <see cref="YoloV11Model"/>; once the detection
/// integration weights are on the validation box, extract a common <c>YoloV11Trunk</c> and delegate both models to
/// it. Kept self-contained here for now so the tested detection path is untouched.</para></summary>
public sealed class YoloV11PoseModel
{
    private readonly YoloConfig _config;

    private readonly ConvBnSilu _layer0;
    private readonly ConvBnSilu _layer1;
    private readonly C3k2 _layer2;
    private readonly ConvBnSilu _layer3;
    private readonly C3k2 _layer4;
    private readonly ConvBnSilu _layer5;
    private readonly C3k2 _layer6;
    private readonly ConvBnSilu _layer7;
    private readonly C3k2 _layer8;
    private readonly Sppf _layer9;
    private readonly C2psa _layer10;
    private readonly C3k2 _layer13;
    private readonly C3k2 _layer16;
    private readonly ConvBnSilu _layer17;
    private readonly C3k2 _layer19;
    private readonly ConvBnSilu _layer20;
    private readonly C3k2 _layer22;
    private readonly PoseHeadV11 _pose;

    public YoloConfig Config => _config;
    public int NumClasses => _config.NumClasses;
    public int NumKeypoints => _config.NumKeypoints;
    public IReadOnlyList<int> Strides => _config.Strides;

    public YoloV11PoseModel(YoloConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.NumKeypoints <= 0)
            throw new ArgumentException("YoloV11PoseModel requires a pose config (NumKeypoints > 0).", nameof(config));
        _config = config;

        int c64 = ScaleCh(config, 64);
        int c128 = ScaleCh(config, 128);
        int c256 = ScaleCh(config, 256);
        int c512 = ScaleCh(config, 512);
        int c1024 = ScaleCh(config, 1024);
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
        _layer13 = new C3k2(c1024 + c512, c512, r2, c3k: false, shortcut: true);
        _layer16 = new C3k2(c512 + c512, c256, r2, c3k: false, shortcut: true);
        _layer17 = new ConvBnSilu(c256, 2, 2, 1, 1);
        _layer19 = new C3k2(c256 + c512, c512, r2, c3k: false, shortcut: true);
        _layer20 = new ConvBnSilu(c512, 2, 2, 1, 1);
        _layer22 = new C3k2(c512 + c1024, c1024, r2, c3k: true, shortcut: true);

        _pose = new PoseHeadV11(
            numClasses: config.NumClasses,
            regMax: config.RegMax,
            numKeypoints: config.NumKeypoints,
            kptDims: config.KptDims,
            inChannels: [c256, c512, c1024],
            strides: [config.Strides[0], config.Strides[1], config.Strides[2]]);
    }

    private static int ScaleCh(YoloConfig config, int baseCh)
    {
        int capped = Math.Min(baseCh, config.MaxChannels);
        int scaled = (int)MathF.Round(capped * config.WidthMultiple);
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
        _layer13.LoadWeights(weights, $"{prefix}.13");
        _layer16.LoadWeights(weights, $"{prefix}.16");
        _layer17.LoadWeights(weights, $"{prefix}.17.conv");
        _layer19.LoadWeights(weights, $"{prefix}.19");
        _layer20.LoadWeights(weights, $"{prefix}.20.conv");
        _layer22.LoadWeights(weights, $"{prefix}.22");
        _pose.LoadWeights(weights, $"{prefix}.23");
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
        foreach (Tensor t in _pose.EnumerateWeights()) yield return t;
    }

    /// <summary>Backbone → neck → pose head. Returns box/class detections <c>[1, 4+nc, A]</c> and keypoints
    /// <c>[1, nk·ndim, A]</c> (input-pixel coords); caller owns both.</summary>
    public (Tensor detections, Tensor keypoints) Forward(IBackend backend, Tensor input)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(input);
        if (input.Shape.Rank != 4 || input.Shape[1] != 3)
            throw new ArgumentException($"YOLO input must be [N, 3, H, W]; got {input.Shape}.", nameof(input));

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
        Tensor x10 = _layer10.Forward(backend, x9); x9.Dispose();

        Tensor x11 = YoloV11Model.Upsample(backend, x10, 2);
        Tensor x12 = YoloV11Model.ConcatChannel(backend, x11, x6); x11.Dispose(); x6.Dispose();
        Tensor x13 = _layer13.Forward(backend, x12); x12.Dispose();

        Tensor x14 = YoloV11Model.Upsample(backend, x13, 2);
        Tensor x15 = YoloV11Model.ConcatChannel(backend, x14, x4); x14.Dispose(); x4.Dispose();
        Tensor x16 = _layer16.Forward(backend, x15); x15.Dispose();

        Tensor x17 = _layer17.Forward(backend, x16);
        Tensor x18 = YoloV11Model.ConcatChannel(backend, x17, x13); x17.Dispose(); x13.Dispose();
        Tensor x19 = _layer19.Forward(backend, x18); x18.Dispose();

        Tensor x20 = _layer20.Forward(backend, x19);
        Tensor x21 = YoloV11Model.ConcatChannel(backend, x20, x10); x20.Dispose(); x10.Dispose();
        Tensor x22 = _layer22.Forward(backend, x21); x21.Dispose();

        (Tensor detections, Tensor keypoints) = _pose.Forward(backend, [x16, x19, x22]);
        x16.Dispose(); x19.Dispose(); x22.Dispose();
        return (detections, keypoints);
    }
}
