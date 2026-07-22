using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.Vision.Annotators;

/// <summary>NormalBAE surface-normal estimator (controlnet_aux <c>NNET</c>, architecture "BN"): a
/// tf_efficientnet_b5_ap encoder (MBConv blocks with squeeze-excitation, TF SAME padding on the stride-2
/// depthwise convs, BatchNorm folded into conv weights at load) whose taps feed a coarse-to-fine decoder —
/// conv/BN upsample blocks build a feature pyramid, then per-pixel 1×1 MLP heads refine the normal at
/// 1/4, 1/2 and 1/1 resolution. Each scale's output is <c>[1,4,·,·]</c>: an L2-normalized xyz normal plus
/// an ELU-mapped kappa concentration. Inference-mode ("test") dense refinement only — the training-time
/// uncertainty-guided sparse sampling is not ported. Input <c>[1,3,H,W]</c> ImageNet-normalized, H and W
/// multiples of 32. Targets the CPU backend (SE pooling and normalization are host glue).</summary>
public sealed unsafe class NormalBaeModel
{
    private const float BnEps = 1e-3f;

    private static readonly (int Repeats, int Kernel, int Stride, int Expand, int OutC)[] Stages =
    [
        (3, 3, 1, 1, 24), (5, 3, 2, 6, 40), (5, 5, 2, 6, 64), (7, 3, 2, 6, 128),
        (7, 5, 1, 6, 176), (9, 5, 2, 6, 304), (3, 3, 1, 6, 512),
    ];
    private static readonly int[] TapStages = [0, 1, 2, 4];

    private readonly NormalBaePreset _preset;
    private Tensor? _stemW, _stemB;
    private Tensor? _headW;
    private readonly List<MbConvBlock[]> _stages = [];
    private Tensor? _conv2W, _conv2B;
    private readonly UpBlock[] _ups = new UpBlock[4];
    private Tensor? _res8W, _res8B;
    private readonly Tensor?[][] _mlpW = new Tensor?[3][];
    private readonly Tensor?[][] _mlpB = new Tensor?[3][];

    public NormalBaePreset Preset => _preset;

    public NormalBaeModel(NormalBaePreset preset)
    {
        ArgumentNullException.ThrowIfNull(preset);
        _preset = preset;
        for (int i = 0; i < 4; i++) _ups[i] = new UpBlock();
        for (int i = 0; i < 3; i++) { _mlpW[i] = new Tensor?[4]; _mlpB[i] = new Tensor?[4]; }
    }

    /// <summary>Loads the flattened <c>scannet.pt</c> state dict (the loader drops the <c>model</c>
    /// envelope). Every conv+BatchNorm pair is folded into a single conv weight/bias here.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        const string e = "encoder.original_model.";
        (_stemW, _stemB) = FoldBn(w, $"{e}conv_stem", $"{e}bn1");
        _headW = Dinov2VisionEncoder.F32(w[$"{e}conv_head.weight"]);

        _stages.Clear();
        int inC = 48;
        for (int s = 0; s < Stages.Length; s++)
        {
            (int repeats, int kernel, int stride, int expand, int outC) = Stages[s];
            MbConvBlock[] blocks = new MbConvBlock[repeats];
            for (int b = 0; b < repeats; b++)
            {
                string p = $"{e}blocks.{s}.{b}.";
                MbConvBlock block = new()
                {
                    Kernel = kernel,
                    Stride = b == 0 ? stride : 1,
                    InC = inC,
                    OutC = outC,
                    MidC = expand == 1 ? inC : inC * expand,
                    HasExpand = expand != 1,
                };
                block.Residual = block.Stride == 1 && block.InC == block.OutC;
                if (block.HasExpand)
                {
                    (block.PwW, block.PwB) = FoldBn(w, $"{p}conv_pw", $"{p}bn1");
                    (block.DwW, block.DwB) = FoldBn(w, $"{p}conv_dw", $"{p}bn2");
                    (block.PwlW, block.PwlB) = FoldBn(w, $"{p}conv_pwl", $"{p}bn3");
                }
                else
                {
                    (block.DwW, block.DwB) = FoldBn(w, $"{p}conv_dw", $"{p}bn1");
                    (block.PwlW, block.PwlB) = FoldBn(w, $"{p}conv_pw", $"{p}bn2");
                }
                block.SeReduceW = Dinov2VisionEncoder.F32(w[$"{p}se.conv_reduce.weight"]);
                block.SeReduceB = Dinov2VisionEncoder.F32(w[$"{p}se.conv_reduce.bias"]);
                block.SeExpandW = Dinov2VisionEncoder.F32(w[$"{p}se.conv_expand.weight"]);
                block.SeExpandB = Dinov2VisionEncoder.F32(w[$"{p}se.conv_expand.bias"]);
                blocks[b] = block;
                inC = outC;
            }
            _stages.Add(blocks);
        }

        const string d = "decoder.";
        _conv2W = Dinov2VisionEncoder.F32(w[$"{d}conv2.weight"]);
        _conv2B = Dinov2VisionEncoder.F32(w[$"{d}conv2.bias"]);
        for (int i = 0; i < 4; i++)
        {
            string p = $"{d}up{i + 1}._net.";
            (_ups[i].Conv1W, _ups[i].Conv1B) = FoldBn(w, $"{p}0", $"{p}1");
            (_ups[i].Conv2W, _ups[i].Conv2B) = FoldBn(w, $"{p}3", $"{p}4");
        }
        _res8W = Dinov2VisionEncoder.F32(w[$"{d}out_conv_res8.weight"]);
        _res8B = Dinov2VisionEncoder.F32(w[$"{d}out_conv_res8.bias"]);
        string[] heads = ["out_conv_res4", "out_conv_res2", "out_conv_res1"];
        for (int hIdx = 0; hIdx < 3; hIdx++)
        {
            for (int l = 0; l < 4; l++)
            {
                Tensor cw = Dinov2VisionEncoder.F32(w[$"{d}{heads[hIdx]}.{l * 2}.weight"]);
                // Conv1d k=1 weight [out, in, 1] → Linear weight [out, in].
                _mlpW[hIdx][l] = cw.Reshape(new TensorShape(cw.Shape[0], cw.Shape[1]));
                _mlpB[hIdx][l] = Dinov2VisionEncoder.F32(w[$"{d}{heads[hIdx]}.{l * 2}.bias"]);
            }
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] flat = [_stemW, _stemB, _headW, _conv2W, _conv2B, _res8W, _res8B];
        foreach (Tensor? t in flat) if (t is not null) yield return t;
        foreach (MbConvBlock[] stage in _stages)
            foreach (MbConvBlock b in stage)
                foreach (Tensor t in b.EnumerateWeights()) yield return t;
        foreach (UpBlock u in _ups) foreach (Tensor t in u.EnumerateWeights()) yield return t;
        for (int h = 0; h < 3; h++)
            for (int l = 0; l < 4; l++)
            {
                if (_mlpW[h][l] is not null) yield return _mlpW[h][l]!;
                if (_mlpB[h][l] is not null) yield return _mlpB[h][l]!;
            }
    }

    /// <summary>Predicts surface normals for ImageNet-normalized pixels <c>[1,3,H,W]</c> (H, W multiples
    /// of 32). Returns <c>[1,4,H,W]</c>: unit xyz normal + kappa. <paramref name="tap"/> receives named
    /// intermediates for the parity harness.</summary>
    public Tensor Forward(IBackend backend, Tensor pixelValues, Action<string, Tensor>? tap = null)
    {
        if (pixelValues.Shape.Rank != 4 || pixelValues.Shape[0] != 1 || pixelValues.Shape[1] != 3)
            throw new ArgumentException($"NormalBAE input must be [1,3,H,W]; got {pixelValues.Shape}.", nameof(pixelValues));
        int h = (int)pixelValues.Shape[2];
        int w = (int)pixelValues.Shape[3];
        if (h % 32 != 0 || w % 32 != 0)
            throw new ArgumentException($"NormalBAE input dimensions must be multiples of 32; got {w}x{h}.", nameof(pixelValues));

        Tensor[] taps = EncodeTaps(backend, pixelValues, h, w, tap);
        Tensor normals = Decode(backend, taps, tap);
        foreach (Tensor t in taps) t.Dispose();
        return normals;
    }

    /// <summary>EfficientNet-B5 trunk; returns the 5 decoder taps (stage0/1/2/4 + conv_head).</summary>
    private Tensor[] EncodeTaps(IBackend backend, Tensor pixels, int h, int w, Action<string, Tensor>? tap)
    {
        Tensor padded = ZeroPad2dSame(pixels, 3, 2, h, w);
        int curH = (h + 1) / 2, curW = (w + 1) / 2;
        Tensor x = new(new TensorShape(1, 48, curH, curW), DType.F32);
        backend.Conv2D(x, padded, _stemW!, _stemB!, 2, 2, 0, 0);
        padded.Dispose();
        Tensor act = new(x.Shape, DType.F32);
        backend.Silu(act, x);
        x.Dispose();
        x = act;

        Tensor[] taps = new Tensor[5];
        int tapIdx = 0;
        for (int s = 0; s < _stages.Count; s++)
        {
            foreach (MbConvBlock block in _stages[s])
            {
                Tensor y = block.Forward(backend, x, curH, curW);
                x.Dispose();
                x = y;
                if (block.Stride == 2) { curH = (curH + 1) / 2; curW = (curW + 1) / 2; }
            }
            if (Array.IndexOf(TapStages, s) >= 0)
            {
                taps[tapIdx] = Clone(x);
                tap?.Invoke($"feat_{tapIdx}", taps[tapIdx]);
                tapIdx++;
            }
        }

        Tensor head = new(new TensorShape(1, 2048, curH, curW), DType.F32);
        backend.Conv2D(head, x, _headW!, null, 1, 1, 0, 0);
        x.Dispose();
        taps[4] = head;
        tap?.Invoke("feat_4", head);
        return taps;
    }

    /// <summary>Decoder: feature pyramid + dense test-mode refinement heads.</summary>
    private Tensor Decode(IBackend backend, Tensor[] taps, Action<string, Tensor>? tap)
    {
        Tensor xd0 = new(taps[4].Shape, DType.F32);
        backend.Conv2D(xd0, taps[4], _conv2W!, _conv2B!, 1, 1, 0, 0);
        tap?.Invoke("xd_0", xd0);
        Tensor xd1 = _ups[0].Forward(backend, xd0, taps[3], 1024);
        tap?.Invoke("xd_1", xd1);
        xd0.Dispose();
        Tensor xd2 = _ups[1].Forward(backend, xd1, taps[2], 512);
        tap?.Invoke("xd_2", xd2);
        xd1.Dispose();
        Tensor xd3 = _ups[2].Forward(backend, xd2, taps[1], 256);
        tap?.Invoke("xd_3", xd3);
        Tensor xd4 = _ups[3].Forward(backend, xd3, taps[0], 128);
        tap?.Invoke("xd_4", xd4);

        Tensor res8 = new(new TensorShape(1, 4, xd2.Shape[2], xd2.Shape[3]), DType.F32);
        backend.Conv2D(res8, xd2, _res8W!, _res8B!, 1, 1, 1, 1);
        NormNormalize(res8);
        tap?.Invoke("out_res8", res8);

        Tensor res4 = RefineHead(backend, xd2, res8, 0);
        tap?.Invoke("out_res4", res4);
        res8.Dispose();
        xd2.Dispose();
        Tensor res2 = RefineHead(backend, xd3, res4, 1);
        tap?.Invoke("out_res2", res2);
        res4.Dispose();
        xd3.Dispose();
        Tensor res1 = RefineHead(backend, xd4, res2, 2);
        tap?.Invoke("out_res1", res1);
        res2.Dispose();
        xd4.Dispose();
        return res1;
    }

    /// <summary>Dense refinement at 2× the previous scale: bilinear-upsample features and previous
    /// prediction, concat, run the per-pixel 1×1 conv MLP, re-normalize.</summary>
    private Tensor RefineHead(IBackend backend, Tensor feat, Tensor prev, int headIdx)
    {
        int outH = (int)feat.Shape[2] * 2, outW = (int)feat.Shape[3] * 2;
        int featC = (int)feat.Shape[1];
        Tensor featUp = new(new TensorShape(1, featC, outH, outW), DType.F32);
        backend.InterpolateBilinear2D(featUp, feat, alignCorners: true);
        Tensor prevUp = new(new TensorShape(1, 4, outH, outW), DType.F32);
        backend.InterpolateBilinear2D(prevUp, prev, alignCorners: true);
        Tensor cat = new(new TensorShape(1, featC + 4, outH, outW), DType.F32);
        backend.Concat(cat, [featUp, prevUp], 1);
        featUp.Dispose();
        prevUp.Dispose();

        // [1, C, H·W] → [H·W, C] rows so the Conv1d-k1 stack becomes a Linear chain.
        long n = (long)outH * outW;
        Tensor rows = new(new TensorShape(n, featC + 4), DType.F32);
        backend.Transpose2D(rows, cat.Reshape(new TensorShape(featC + 4, n)), featC + 4, (int)n);
        cat.Dispose();

        Tensor cur = rows;
        for (int l = 0; l < 4; l++)
        {
            Tensor next = new(new TensorShape(n, _mlpW[headIdx][l]!.Shape[0]), DType.F32);
            backend.Linear(next, cur, _mlpW[headIdx][l]!, _mlpB[headIdx][l]!);
            cur.Dispose();
            if (l < 3)
            {
                Tensor relu = new(next.Shape, DType.F32);
                backend.LeakyRelu(relu, next, 0f);
                next.Dispose();
                cur = relu;
            }
            else
            {
                cur = next;
            }
        }

        Tensor output2d = new(new TensorShape(4, n), DType.F32);
        backend.Transpose2D(output2d, cur, (int)n, 4);
        cur.Dispose();
        Tensor output = output2d.Reshape(new TensorShape(1, 4, outH, outW));
        NormNormalize(output);
        return output;
    }

    /// <summary>controlnet_aux <c>norm_normalize</c> in place on <c>[1,4,h,w]</c>: L2-normalize xyz,
    /// kappa → <c>elu(kappa) + 1.01</c>. Host loop — a few calls per forward on the output planes.</summary>
    public static void NormNormalize(Tensor normals)
    {
        if (normals.Shape.Rank != 4 || normals.Shape[1] != 4)
            throw new ArgumentException($"norm_normalize expects [B,4,h,w]; got {normals.Shape}.", nameof(normals));
        long plane = normals.Shape[2] * normals.Shape[3];
        float* p = (float*)normals.DataPointer;
        for (long i = 0; i < plane; i++)
        {
            float x = p[i], y = p[plane + i], z = p[2 * plane + i], k = p[3 * plane + i];
            float norm = MathF.Sqrt(x * x + y * y + z * z) + 1e-10f;
            p[i] = x / norm;
            p[plane + i] = y / norm;
            p[2 * plane + i] = z / norm;
            p[3 * plane + i] = (k > 0f ? k : MathF.Exp(k) - 1f) + 1.01f;
        }
    }

    /// <summary>TF "SAME" asymmetric zero pad for a stride-<paramref name="stride"/> conv (pads more on
    /// bottom/right, per-dim <c>total = (ceil(in/s)−1)·s + k − in</c>).</summary>
    internal static Tensor ZeroPad2dSame(Tensor input, int kernel, int stride, int h, int w)
    {
        int padH = Math.Max(0, ((h + stride - 1) / stride - 1) * stride + kernel - h);
        int padW = Math.Max(0, ((w + stride - 1) / stride - 1) * stride + kernel - w);
        int top = padH / 2, left = padW / 2;
        int n = (int)input.Shape[0], c = (int)input.Shape[1];
        Tensor output = new(new TensorShape(n, c, h + padH, w + padW), DType.F32);
        float* src = (float*)input.DataPointer;
        float* dst = (float*)output.DataPointer;
        int oh = h + padH, ow = w + padW;
        new Span<float>(dst, checked((int)output.ElementCount)).Clear();
        for (long plane = 0; plane < (long)n * c; plane++)
            for (int y = 0; y < h; y++)
                Buffer.MemoryCopy(
                    src + (plane * h + y) * w,
                    dst + (plane * oh + top + y) * ow + left,
                    (long)w * sizeof(float), (long)w * sizeof(float));
        return output;
    }

    private static Tensor Clone(Tensor t)
    {
        Tensor copy = new(t.Shape, DType.F32);
        Buffer.MemoryCopy((void*)t.DataPointer, (void*)copy.DataPointer, t.ElementCount * 4, t.ElementCount * 4);
        return copy;
    }

    /// <summary>Folds inference BatchNorm (eps 1e-3, the tf_ variants' value) into the preceding conv.</summary>
    private static (Tensor W, Tensor B) FoldBn(IReadOnlyDictionary<string, Tensor> w, string convKey, string bnKey)
        => ConvBnFold.Fold(w, convKey, bnKey, BnEps);

    /// <summary>One MBConv block (or the stage-0 depthwise-separable variant when
    /// <see cref="HasExpand"/> is false), BN pre-folded.</summary>
    private sealed class MbConvBlock
    {
        public Tensor? PwW, PwB;
        public Tensor? DwW, DwB;
        public Tensor? SeReduceW, SeReduceB, SeExpandW, SeExpandB;
        public Tensor? PwlW, PwlB;
        public int Kernel, Stride, InC, MidC, OutC;
        public bool HasExpand, Residual;

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] all = [PwW, PwB, DwW, DwB, SeReduceW, SeReduceB, SeExpandW, SeExpandB, PwlW, PwlB];
            foreach (Tensor? t in all) if (t is not null) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor input, int h, int w)
        {
            Tensor x = input;
            if (HasExpand)
            {
                Tensor pw = new(new TensorShape(1, MidC, h, w), DType.F32);
                backend.Conv2D(pw, x, PwW!, PwB!, 1, 1, 0, 0);
                Tensor act = new(pw.Shape, DType.F32);
                backend.Silu(act, pw);
                pw.Dispose();
                x = act;
            }

            int outH = h, outW = w;
            Tensor dw;
            if (Stride == 2)
            {
                outH = (h + 1) / 2; outW = (w + 1) / 2;
                Tensor padded = ZeroPad2dSame(x, Kernel, 2, h, w);
                if (!ReferenceEquals(x, input)) x.Dispose();
                dw = new Tensor(new TensorShape(1, MidC, outH, outW), DType.F32);
                backend.Conv2dDepthwise(dw, padded, DwW!, DwB!, 2, 2, 0, 0);
                padded.Dispose();
            }
            else
            {
                dw = new Tensor(new TensorShape(1, MidC, h, w), DType.F32);
                backend.Conv2dDepthwise(dw, x, DwW!, DwB!, 1, 1, Kernel / 2, Kernel / 2);
                if (!ReferenceEquals(x, input)) x.Dispose();
            }
            Tensor dwAct = new(dw.Shape, DType.F32);
            backend.Silu(dwAct, dw);
            dw.Dispose();

            Tensor gated = SqueezeExcite(backend, dwAct, outH, outW);
            dwAct.Dispose();

            Tensor proj = new(new TensorShape(1, OutC, outH, outW), DType.F32);
            backend.Conv2D(proj, gated, PwlW!, PwlB!, 1, 1, 0, 0);
            gated.Dispose();

            if (!Residual) return proj;
            Tensor sum = new(proj.Shape, DType.F32);
            backend.Add(sum, proj, input);
            proj.Dispose();
            return sum;
        }

        /// <summary>Squeeze-excitation: global-average pool (host reduce), 1×1 reduce → SiLU → 1×1 expand
        /// → sigmoid, channel gate applied via <c>MaskRows</c> on a <c>[C, H·W]</c> view.</summary>
        private Tensor SqueezeExcite(IBackend backend, Tensor x, int h, int w)
        {
            int c = (int)x.Shape[1];
            long plane = (long)h * w;
            Tensor pooled = new(new TensorShape(1, c, 1, 1), DType.F32);
            float* src = (float*)x.DataPointer;
            float* dst = (float*)pooled.DataPointer;
            for (int ch = 0; ch < c; ch++)
            {
                double sum = 0;
                float* p = src + ch * plane;
                for (long i = 0; i < plane; i++) sum += p[i];
                dst[ch] = (float)(sum / plane);
            }

            int reduced = (int)SeReduceW!.Shape[0];
            Tensor red = new(new TensorShape(1, reduced, 1, 1), DType.F32);
            backend.Conv2D(red, pooled, SeReduceW!, SeReduceB!, 1, 1, 0, 0);
            pooled.Dispose();
            Tensor redAct = new(red.Shape, DType.F32);
            backend.Silu(redAct, red);
            red.Dispose();
            Tensor exp = new(new TensorShape(1, c, 1, 1), DType.F32);
            backend.Conv2D(exp, redAct, SeExpandW!, SeExpandB!, 1, 1, 0, 0);
            redAct.Dispose();
            Tensor gate = new(exp.Shape, DType.F32);
            backend.Sigmoid(gate, exp);
            exp.Dispose();

            // Allocate the output already in the 2-D shape MaskRows writes: on CUDA, GpuTransferHelper's
            // activation cache is keyed by the exact Tensor object the op writes into. Reshaping x.Shape
            // *after* allocating and passing that reshaped view as the output arg would register the GPU
            // write under the view's identity, orphaning the 4-D object this method returns — the caller
            // would then read that object's untouched (zeroed) host buffer. Reshape only on the way out,
            // which forces a DataPointer sync on the object the cache actually knows about.
            Tensor output2d = new(new TensorShape(c, plane), DType.F32);
            backend.MaskRows(output2d, x.Reshape(new TensorShape(c, plane)), gate.Reshape(new TensorShape(c)));
            gate.Dispose();
            return output2d.Reshape(x.Shape);
        }
    }

    /// <summary>Decoder up block: bilinear (align_corners=True) to the skip's size, channel concat, then
    /// two folded conv3×3+LeakyReLU(0.01).</summary>
    private sealed class UpBlock
    {
        public Tensor? Conv1W, Conv1B, Conv2W, Conv2B;

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] all = [Conv1W, Conv1B, Conv2W, Conv2B];
            foreach (Tensor? t in all) if (t is not null) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor x, Tensor skip, int outC)
        {
            int outH = (int)skip.Shape[2], outW = (int)skip.Shape[3];
            Tensor up = new(new TensorShape(1, x.Shape[1], outH, outW), DType.F32);
            backend.InterpolateBilinear2D(up, x, alignCorners: true);
            Tensor cat = new(new TensorShape(1, x.Shape[1] + skip.Shape[1], outH, outW), DType.F32);
            backend.Concat(cat, [up, skip], 1);
            up.Dispose();

            Tensor c1 = new(new TensorShape(1, outC, outH, outW), DType.F32);
            backend.Conv2D(c1, cat, Conv1W!, Conv1B!, 1, 1, 1, 1);
            cat.Dispose();
            Tensor a1 = new(c1.Shape, DType.F32);
            backend.LeakyRelu(a1, c1, 0.01f);
            c1.Dispose();
            Tensor c2 = new(a1.Shape, DType.F32);
            backend.Conv2D(c2, a1, Conv2W!, Conv2B!, 1, 1, 1, 1);
            a1.Dispose();
            Tensor a2 = new(c2.Shape, DType.F32);
            backend.LeakyRelu(a2, c2, 0.01f);
            c2.Dispose();
            return a2;
        }
    }
}
