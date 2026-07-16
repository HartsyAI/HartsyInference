using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.Vision.Annotators;

/// <summary>The lineart annotator's image-to-sketch <c>Generator</c> (comfyui_controlnet_aux
/// <c>LineartDetector</c>): a compact ResNet-style UNet — 7×7 stem, two stride-2 downsamples
/// (64→128→256), 3 reflection-padded residual blocks, two ConvTranspose upsamples back to 64, and a 7×7
/// sigmoid head producing a single-channel sketch in [0,1] (white background, dark lines; the ControlNet
/// conditioning form is the inverse — see <see cref="LineartPreprocessor"/>). All normalization is
/// affine-free InstanceNorm, realized as per-channel GroupNorm. Input [1,3,H,W] in [0,1]; H and W must be
/// multiples of 4 so the down/up path round-trips exactly.</summary>
public sealed unsafe class LineartGenerator
{
    private const float InstanceNormEps = 1e-5f;

    private readonly LineartPreset _preset;
    private Tensor? _stemW, _stemB;
    private Tensor? _down1W, _down1B, _down2W, _down2B;
    private readonly Tensor?[] _resAW;
    private readonly Tensor?[] _resAB;
    private readonly Tensor?[] _resBW;
    private readonly Tensor?[] _resBB;
    private Tensor? _up1W, _up1B, _up2W, _up2B;
    private Tensor? _headW, _headB;
    private readonly Dictionary<int, (Tensor Ones, Tensor Zeros)> _normAffine = [];

    public LineartPreset Preset => _preset;

    public LineartGenerator(LineartPreset preset)
    {
        ArgumentNullException.ThrowIfNull(preset);
        _preset = preset;
        _resAW = new Tensor?[preset.ResidualBlocks];
        _resAB = new Tensor?[preset.ResidualBlocks];
        _resBW = new Tensor?[preset.ResidualBlocks];
        _resBB = new Tensor?[preset.ResidualBlocks];
    }

    /// <summary>Loads the official <c>sk_model.pth</c> / <c>sk_model2.pth</c> state dict
    /// (keys <c>model0.1</c>, <c>model1.{0,3}</c>, <c>model2.{i}.conv_block.{1,5}</c>,
    /// <c>model3.{0,3}</c>, <c>model4.1</c> — InstanceNorm layers carry no parameters).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _stemW = Dinov2VisionEncoder.F32(w["model0.1.weight"]);
        _stemB = Dinov2VisionEncoder.F32(w["model0.1.bias"]);
        _down1W = Dinov2VisionEncoder.F32(w["model1.0.weight"]);
        _down1B = Dinov2VisionEncoder.F32(w["model1.0.bias"]);
        _down2W = Dinov2VisionEncoder.F32(w["model1.3.weight"]);
        _down2B = Dinov2VisionEncoder.F32(w["model1.3.bias"]);
        for (int i = 0; i < _preset.ResidualBlocks; i++)
        {
            _resAW[i] = Dinov2VisionEncoder.F32(w[$"model2.{i}.conv_block.1.weight"]);
            _resAB[i] = Dinov2VisionEncoder.F32(w[$"model2.{i}.conv_block.1.bias"]);
            _resBW[i] = Dinov2VisionEncoder.F32(w[$"model2.{i}.conv_block.5.weight"]);
            _resBB[i] = Dinov2VisionEncoder.F32(w[$"model2.{i}.conv_block.5.bias"]);
        }
        _up1W = Dinov2VisionEncoder.F32(w["model3.0.weight"]);
        _up1B = Dinov2VisionEncoder.F32(w["model3.0.bias"]);
        _up2W = Dinov2VisionEncoder.F32(w["model3.3.weight"]);
        _up2B = Dinov2VisionEncoder.F32(w["model3.3.bias"]);
        _headW = Dinov2VisionEncoder.F32(w["model4.1.weight"]);
        _headB = Dinov2VisionEncoder.F32(w["model4.1.bias"]);

        foreach (int width in (int[])[64, 128, 256])
        {
            if (_normAffine.ContainsKey(width)) continue;
            Tensor ones = new(new TensorShape(width), DType.F32);
            Tensor zeros = new(new TensorShape(width), DType.F32);
            float* op = (float*)ones.DataPointer;
            float* zp = (float*)zeros.DataPointer;
            for (int i = 0; i < width; i++) { op[i] = 1f; zp[i] = 0f; }
            _normAffine[width] = (ones, zeros);
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] flat = [_stemW, _stemB, _down1W, _down1B, _down2W, _down2B, _up1W, _up1B, _up2W, _up2B, _headW, _headB];
        foreach (Tensor? t in flat) if (t is not null) yield return t;
        for (int i = 0; i < _preset.ResidualBlocks; i++)
        {
            if (_resAW[i] is not null) yield return _resAW[i]!;
            if (_resAB[i] is not null) yield return _resAB[i]!;
            if (_resBW[i] is not null) yield return _resBW[i]!;
            if (_resBB[i] is not null) yield return _resBB[i]!;
        }
        foreach ((Tensor ones, Tensor zeros) in _normAffine.Values) { yield return ones; yield return zeros; }
    }

    /// <summary>Sketch prediction for pixels <c>[1,3,H,W]</c> in [0,1] (H, W multiples of 4). Returns the
    /// sigmoid output <c>[1,1,H,W]</c>. <paramref name="tap"/> receives named intermediates for the parity
    /// harness.</summary>
    public Tensor Forward(IBackend backend, Tensor pixelValues, Action<string, Tensor>? tap = null)
    {
        if (pixelValues.Shape.Rank != 4 || pixelValues.Shape[0] != 1 || pixelValues.Shape[1] != 3)
            throw new ArgumentException($"Lineart input must be [1,3,H,W]; got {pixelValues.Shape}.", nameof(pixelValues));
        int h = (int)pixelValues.Shape[2];
        int w = (int)pixelValues.Shape[3];
        if (h % 4 != 0 || w % 4 != 0)
            throw new ArgumentException($"Lineart input dimensions must be multiples of 4; got {w}x{h}.", nameof(pixelValues));

        // model0: ReflectionPad(3) → 7×7 conv → InstanceNorm → ReLU.
        Tensor x = ConvNormRelu(backend, pixelValues, _stemW!, _stemB!, 64, 1, reflectPad: 3, h, w);
        tap?.Invoke("m0", x);

        // model1: two stride-2 3×3 downsamples (zero pad 1).
        int h2 = h / 2, w2 = w / 2, h4 = h / 4, w4 = w / 4;
        Tensor d1 = ConvNormRelu(backend, x, _down1W!, _down1B!, 128, 2, reflectPad: 0, h2, w2);
        x.Dispose();
        Tensor d2 = ConvNormRelu(backend, d1, _down2W!, _down2B!, 256, 2, reflectPad: 0, h4, w4);
        d1.Dispose();
        x = d2;
        tap?.Invoke("m1", x);

        // model2: reflection-padded residual blocks (conv-IN-ReLU-conv-IN, + skip).
        for (int i = 0; i < _preset.ResidualBlocks; i++)
        {
            Tensor a = ConvNormRelu(backend, x, _resAW[i]!, _resAB[i]!, 256, 1, reflectPad: 1, h4, w4);
            Tensor b = ConvNorm(backend, a, _resBW[i]!, _resBB[i]!, 256, 1, reflectPad: 1, h4, w4);
            a.Dispose();
            Tensor sum = new(x.Shape, DType.F32);
            backend.Add(sum, x, b);
            b.Dispose();
            x.Dispose();
            x = sum;
        }
        tap?.Invoke("m2", x);

        // model3: two ConvTranspose(3, s2, p1, op1) upsamples.
        Tensor u1 = new(new TensorShape(1, 128, h2, w2), DType.F32);
        backend.ConvTranspose2d(u1, x, _up1W!, _up1B!, 2, 2, 1, 1);
        x.Dispose();
        Tensor u1n = NormRelu(backend, u1, 128);
        u1.Dispose();
        Tensor u2 = new(new TensorShape(1, 64, h, w), DType.F32);
        backend.ConvTranspose2d(u2, u1n, _up2W!, _up2B!, 2, 2, 1, 1);
        u1n.Dispose();
        Tensor u2n = NormRelu(backend, u2, 64);
        u2.Dispose();
        x = u2n;
        tap?.Invoke("m3", x);

        // model4: ReflectionPad(3) → 7×7 conv → sigmoid.
        Tensor padded = ReflectionPad2d(x, 3);
        x.Dispose();
        Tensor logits = new(new TensorShape(1, 1, h, w), DType.F32);
        backend.Conv2D(logits, padded, _headW!, _headB!, 1, 1, 0, 0);
        padded.Dispose();
        Tensor line = new(logits.Shape, DType.F32);
        backend.Sigmoid(line, logits);
        logits.Dispose();
        return line;
    }

    /// <summary>PyTorch <c>nn.ReflectionPad2d</c>: mirrors <paramref name="pad"/> rows/columns without
    /// repeating the border pixel. Host copy — runs a handful of times per generation on host-resident
    /// activations (this model targets the CPU backend).</summary>
    public static Tensor ReflectionPad2d(Tensor input, int pad)
    {
        if (input.Shape.Rank != 4)
            throw new ArgumentException($"ReflectionPad2d requires [N,C,H,W]; got {input.Shape}.", nameof(input));
        int n = (int)input.Shape[0], c = (int)input.Shape[1];
        int h = (int)input.Shape[2], w = (int)input.Shape[3];
        if (pad >= h || pad >= w)
            throw new ArgumentException($"Reflection pad {pad} must be smaller than both spatial dims ({h}x{w}).", nameof(pad));

        int oh = h + 2 * pad, ow = w + 2 * pad;
        Tensor output = new(new TensorShape(n, c, oh, ow), DType.F32);
        float* src = (float*)input.DataPointer;
        float* dst = (float*)output.DataPointer;
        for (long plane = 0; plane < (long)n * c; plane++)
        {
            float* sp = src + plane * h * w;
            float* dp = dst + plane * oh * ow;
            for (int oy = 0; oy < oh; oy++)
            {
                int sy = Reflect(oy - pad, h);
                for (int ox = 0; ox < ow; ox++)
                    dp[(long)oy * ow + ox] = sp[(long)sy * w + Reflect(ox - pad, w)];
            }
        }
        return output;
    }

    private static int Reflect(int i, int n)
    {
        if (i < 0) return -i;
        if (i >= n) return 2 * (n - 1) - i;
        return i;
    }

    private Tensor ConvNormRelu(IBackend backend, Tensor input, Tensor weight, Tensor bias, int outC, int stride, int reflectPad, int outH, int outW)
    {
        Tensor conv = ConvNorm(backend, input, weight, bias, outC, stride, reflectPad, outH, outW);
        Tensor relu = new(conv.Shape, DType.F32);
        backend.LeakyRelu(relu, conv, 0f);
        conv.Dispose();
        return relu;
    }

    private Tensor ConvNorm(IBackend backend, Tensor input, Tensor weight, Tensor bias, int outC, int stride, int reflectPad, int outH, int outW)
    {
        Tensor conv = new(new TensorShape(1, outC, outH, outW), DType.F32);
        if (reflectPad > 0)
        {
            Tensor padded = ReflectionPad2d(input, reflectPad);
            backend.Conv2D(conv, padded, weight, bias, stride, stride, 0, 0);
            padded.Dispose();
        }
        else
        {
            backend.Conv2D(conv, input, weight, bias, stride, stride, 1, 1);
        }
        Tensor normed = new(conv.Shape, DType.F32);
        (Tensor ones, Tensor zeros) = _normAffine[outC];
        backend.GroupNorm(normed, conv, ones, zeros, outC, InstanceNormEps);
        conv.Dispose();
        return normed;
    }

    private Tensor NormRelu(IBackend backend, Tensor input, int channels)
    {
        Tensor normed = new(input.Shape, DType.F32);
        (Tensor ones, Tensor zeros) = _normAffine[channels];
        backend.GroupNorm(normed, input, ones, zeros, channels, InstanceNormEps);
        Tensor relu = new(normed.Shape, DType.F32);
        backend.LeakyRelu(relu, normed, 0f);
        normed.Dispose();
        return relu;
    }
}
