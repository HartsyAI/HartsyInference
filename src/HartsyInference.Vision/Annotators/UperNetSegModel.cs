using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.Vision.Annotators;

/// <summary>UperNet semantic-segmentation head on a ConvNeXt-Small backbone (transformers
/// <c>UperNetForSemanticSegmentation</c>, checkpoint <c>openmmlab/upernet-convnext-small</c>, ADE20K
/// 150 classes) — the reference annotator for <c>control_v11p_sd15_seg</c> conditioning. Backbone:
/// 4×4/4 patch stem + 4 stages (depths 3/3/27/3, dims 96/192/384/768) of ConvNeXt blocks (7×7 depthwise
/// conv → channels-last LayerNorm → 4× MLP with exact-erf GELU → layer-scale, folded into the second
/// Linear at load → residual), each stage output LayerNormed for the head. Head: PSP pyramid pooling
/// (scales 1/2/3/6) + FPN with 512 channels, every conv+BatchNorm pair folded at load, classifier to 150
/// logits at 1/4 resolution, bilinearly upsampled (align_corners=False) to the input size. Auxiliary FCN
/// head is training-only and not loaded.</summary>
public sealed unsafe class UperNetSegModel
{
    /// <summary>ADE20K class count (head output channels).</summary>
    public const int NumClasses = 150;

    /// <summary>UperNet FPN/PSP channel width.</summary>
    public const int HeadChannels = 512;

    private const float LnEps = 1e-6f;
    private const float BnEps = 1e-5f;
    private static readonly int[] Depths = [3, 3, 27, 3];
    private static readonly int[] Dims = [96, 192, 384, 768];
    private static readonly int[] PoolScales = [1, 2, 3, 6];

    private Tensor? _stemConvW, _stemConvB, _stemLnW, _stemLnB;
    private readonly Tensor?[] _dsLnW = new Tensor?[4];
    private readonly Tensor?[] _dsLnB = new Tensor?[4];
    private readonly Tensor?[] _dsConvW = new Tensor?[4];
    private readonly Tensor?[] _dsConvB = new Tensor?[4];
    private readonly ConvNextBlock[][] _blocks = new ConvNextBlock[4][];
    private readonly Tensor?[] _featNormW = new Tensor?[4];
    private readonly Tensor?[] _featNormB = new Tensor?[4];
    private readonly Tensor?[] _pspW = new Tensor?[4];
    private readonly Tensor?[] _pspB = new Tensor?[4];
    private Tensor? _bottleneckW, _bottleneckB;
    private readonly Tensor?[] _lateralW = new Tensor?[3];
    private readonly Tensor?[] _lateralB = new Tensor?[3];
    private readonly Tensor?[] _fpnW = new Tensor?[3];
    private readonly Tensor?[] _fpnB = new Tensor?[3];
    private Tensor? _fpnBottleneckW, _fpnBottleneckB;
    private Tensor? _classifierW, _classifierB;

    public UperNetSegModel()
    {
        for (int s = 0; s < 4; s++) _blocks[s] = new ConvNextBlock[Depths[s]];
    }

    /// <summary>Loads the <c>openmmlab/upernet-convnext-small</c> state dict (transformers key layout:
    /// <c>backbone.*</c> + <c>decode_head.*</c>). BatchNorms and layer-scale gammas are folded here;
    /// <c>auxiliary_head.*</c> keys are ignored.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _stemConvW = Dinov2VisionEncoder.F32(w["backbone.embeddings.patch_embeddings.weight"]);
        _stemConvB = Dinov2VisionEncoder.F32(w["backbone.embeddings.patch_embeddings.bias"]);
        _stemLnW = Dinov2VisionEncoder.F32(w["backbone.embeddings.layernorm.weight"]);
        _stemLnB = Dinov2VisionEncoder.F32(w["backbone.embeddings.layernorm.bias"]);

        for (int s = 0; s < 4; s++)
        {
            if (s > 0)
            {
                string ds = $"backbone.encoder.stages.{s}.downsampling_layer.";
                _dsLnW[s] = Dinov2VisionEncoder.F32(w[$"{ds}0.weight"]);
                _dsLnB[s] = Dinov2VisionEncoder.F32(w[$"{ds}0.bias"]);
                _dsConvW[s] = Dinov2VisionEncoder.F32(w[$"{ds}1.weight"]);
                _dsConvB[s] = Dinov2VisionEncoder.F32(w[$"{ds}1.bias"]);
            }
            for (int b = 0; b < Depths[s]; b++)
            {
                string p = $"backbone.encoder.stages.{s}.layers.{b}.";
                ConvNextBlock block = new()
                {
                    DwW = Dinov2VisionEncoder.F32(w[$"{p}dwconv.weight"]),
                    DwB = Dinov2VisionEncoder.F32(w[$"{p}dwconv.bias"]),
                    LnW = Dinov2VisionEncoder.F32(w[$"{p}layernorm.weight"]),
                    LnB = Dinov2VisionEncoder.F32(w[$"{p}layernorm.bias"]),
                    Pw1W = Dinov2VisionEncoder.F32(w[$"{p}pwconv1.weight"]),
                    Pw1B = Dinov2VisionEncoder.F32(w[$"{p}pwconv1.bias"]),
                };
                (block.Pw2W, block.Pw2B) = FoldLayerScale(
                    Dinov2VisionEncoder.F32(w[$"{p}pwconv2.weight"]),
                    Dinov2VisionEncoder.F32(w[$"{p}pwconv2.bias"]),
                    Dinov2VisionEncoder.F32(w[$"{p}layer_scale_parameter"]));
                _blocks[s][b] = block;
            }
            _featNormW[s] = Dinov2VisionEncoder.F32(w[$"backbone.hidden_states_norms.stage{s + 1}.weight"]);
            _featNormB[s] = Dinov2VisionEncoder.F32(w[$"backbone.hidden_states_norms.stage{s + 1}.bias"]);
        }

        for (int k = 0; k < 4; k++)
            (_pspW[k], _pspB[k]) = ConvBnFold.Fold(w, $"decode_head.psp_modules.{k}.1.conv", $"decode_head.psp_modules.{k}.1.batch_norm", BnEps);
        (_bottleneckW, _bottleneckB) = ConvBnFold.Fold(w, "decode_head.bottleneck.conv", "decode_head.bottleneck.batch_norm", BnEps);
        for (int i = 0; i < 3; i++)
        {
            (_lateralW[i], _lateralB[i]) = ConvBnFold.Fold(w, $"decode_head.lateral_convs.{i}.conv", $"decode_head.lateral_convs.{i}.batch_norm", BnEps);
            (_fpnW[i], _fpnB[i]) = ConvBnFold.Fold(w, $"decode_head.fpn_convs.{i}.conv", $"decode_head.fpn_convs.{i}.batch_norm", BnEps);
        }
        (_fpnBottleneckW, _fpnBottleneckB) = ConvBnFold.Fold(w, "decode_head.fpn_bottleneck.conv", "decode_head.fpn_bottleneck.batch_norm", BnEps);
        _classifierW = Dinov2VisionEncoder.F32(w["decode_head.classifier.weight"]);
        _classifierB = Dinov2VisionEncoder.F32(w["decode_head.classifier.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] flat =
        [
            _stemConvW, _stemConvB, _stemLnW, _stemLnB, _bottleneckW, _bottleneckB,
            _fpnBottleneckW, _fpnBottleneckB, _classifierW, _classifierB,
        ];
        foreach (Tensor? t in flat) if (t is not null) yield return t;
        for (int s = 0; s < 4; s++)
        {
            Tensor?[] stage = [_dsLnW[s], _dsLnB[s], _dsConvW[s], _dsConvB[s], _featNormW[s], _featNormB[s], _pspW[s], _pspB[s]];
            foreach (Tensor? t in stage) if (t is not null) yield return t;
            foreach (ConvNextBlock block in _blocks[s])
                if (block is not null)
                    foreach (Tensor t in block.EnumerateWeights()) yield return t;
        }
        for (int i = 0; i < 3; i++)
        {
            Tensor?[] fpn = [_lateralW[i], _lateralB[i], _fpnW[i], _fpnB[i]];
            foreach (Tensor? t in fpn) if (t is not null) yield return t;
        }
    }

    /// <summary>Computes per-pixel class logits <c>[1,150,H,W]</c> for ImageNet-normalized pixels
    /// <c>[1,3,H,W]</c> (H, W multiples of 32; the reference pipeline runs at 512×512). Matches
    /// transformers' <c>outputs.logits</c> (head logits at 1/4 resolution bilinearly upsampled,
    /// align_corners=False). <paramref name="tap"/> receives named intermediates for the parity harness.</summary>
    public Tensor Forward(IBackend backend, Tensor pixelValues, Action<string, Tensor>? tap = null)
    {
        if (pixelValues.Shape.Rank != 4 || pixelValues.Shape[0] != 1 || pixelValues.Shape[1] != 3)
            throw new ArgumentException($"UperNet input must be [1,3,H,W]; got {pixelValues.Shape}.", nameof(pixelValues));
        int fullH = (int)pixelValues.Shape[2];
        int fullW = (int)pixelValues.Shape[3];
        if (fullH % 32 != 0 || fullW % 32 != 0)
            throw new ArgumentException($"UperNet input dimensions must be multiples of 32; got {fullW}x{fullH}.", nameof(pixelValues));

        Tensor[] feats = Backbone(backend, pixelValues, fullH, fullW, tap);
        Tensor logits = DecodeHead(backend, feats, fullH, fullW, tap);
        foreach (Tensor f in feats) f.Dispose();
        return logits;
    }

    /// <summary>ConvNeXt trunk; returns the 4 stage features after the per-stage hidden-state LayerNorms.</summary>
    private Tensor[] Backbone(IBackend backend, Tensor pixels, int fullH, int fullW, Action<string, Tensor>? tap)
    {
        int h = fullH / 4, w = fullW / 4;
        Tensor x = new(new TensorShape(1, Dims[0], h, w), DType.F32);
        backend.Conv2D(x, pixels, _stemConvW!, _stemConvB!, 4, 4, 0, 0);
        Tensor stemLn = LayerNormChannelsFirst(backend, x, _stemLnW!, _stemLnB!);
        x.Dispose();
        x = stemLn;

        Tensor[] feats = new Tensor[4];
        for (int s = 0; s < 4; s++)
        {
            if (s > 0)
            {
                Tensor dsLn = LayerNormChannelsFirst(backend, x, _dsLnW[s]!, _dsLnB[s]!);
                x.Dispose();
                h /= 2; w /= 2;
                Tensor ds = new(new TensorShape(1, Dims[s], h, w), DType.F32);
                backend.Conv2D(ds, dsLn, _dsConvW[s]!, _dsConvB[s]!, 2, 2, 0, 0);
                dsLn.Dispose();
                x = ds;
            }
            foreach (ConvNextBlock block in _blocks[s])
            {
                Tensor y = block.Forward(backend, x, Dims[s], h, w);
                x.Dispose();
                x = y;
            }
            feats[s] = LayerNormChannelsFirst(backend, x, _featNormW[s]!, _featNormB[s]!);
            tap?.Invoke($"feat_{s}", feats[s]);
        }
        x.Dispose();
        return feats;
    }

    /// <summary>UperNet head: PSP pooling on the deepest feature, FPN top-down fusion, classifier,
    /// final bilinear upsample to the input resolution.</summary>
    private Tensor DecodeHead(IBackend backend, Tensor[] feats, int fullH, int fullW, Action<string, Tensor>? tap)
    {
        Tensor[] laterals = new Tensor[4];
        for (int i = 0; i < 3; i++)
            laterals[i] = ConvBnRelu(backend, feats[i], _lateralW[i]!, _lateralB[i]!, 0);
        laterals[3] = PspForward(backend, feats[3], tap);

        for (int i = 3; i > 0; i--)
        {
            Tensor up = new(laterals[i - 1].Shape, DType.F32);
            backend.InterpolateBilinear2D(up, laterals[i], alignCorners: false);
            Tensor sum = new(laterals[i - 1].Shape, DType.F32);
            backend.Add(sum, laterals[i - 1], up);
            up.Dispose();
            laterals[i - 1].Dispose();
            laterals[i - 1] = sum;
        }

        Tensor[] fpnOuts = new Tensor[4];
        for (int i = 0; i < 3; i++)
        {
            fpnOuts[i] = ConvBnRelu(backend, laterals[i], _fpnW[i]!, _fpnB[i]!, 1);
            laterals[i].Dispose();
        }
        fpnOuts[3] = laterals[3];

        int h4 = (int)fpnOuts[0].Shape[2], w4 = (int)fpnOuts[0].Shape[3];
        for (int i = 1; i < 4; i++)
        {
            Tensor up = new(new TensorShape(1, HeadChannels, h4, w4), DType.F32);
            backend.InterpolateBilinear2D(up, fpnOuts[i], alignCorners: false);
            fpnOuts[i].Dispose();
            fpnOuts[i] = up;
        }
        Tensor cat = new(new TensorShape(1, 4 * HeadChannels, h4, w4), DType.F32);
        backend.Concat(cat, fpnOuts, 1);
        foreach (Tensor f in fpnOuts) f.Dispose();

        Tensor fused = ConvBnRelu(backend, cat, _fpnBottleneckW!, _fpnBottleneckB!, 1);
        cat.Dispose();
        tap?.Invoke("fpn_out", fused);

        Tensor logitsQ = new(new TensorShape(1, NumClasses, h4, w4), DType.F32);
        backend.Conv2D(logitsQ, fused, _classifierW!, _classifierB!, 1, 1, 0, 0);
        fused.Dispose();
        tap?.Invoke("logits_q", logitsQ);

        Tensor logits = new(new TensorShape(1, NumClasses, fullH, fullW), DType.F32);
        backend.InterpolateBilinear2D(logits, logitsQ, alignCorners: false);
        logitsQ.Dispose();
        return logits;
    }

    /// <summary>PSP module on the 1/32 feature: adaptive-average pools at scales 1/2/3/6, 1×1 conv+ReLU,
    /// bilinear upsample back, concat with the input, 3×3 bottleneck.</summary>
    private Tensor PspForward(IBackend backend, Tensor x4, Action<string, Tensor>? tap)
    {
        int c = (int)x4.Shape[1];
        int h = (int)x4.Shape[2], w = (int)x4.Shape[3];
        Tensor[] pieces = new Tensor[5];
        pieces[0] = x4;
        for (int k = 0; k < 4; k++)
        {
            Tensor pooled = AdaptiveAvgPool(x4, PoolScales[k]);
            Tensor conv = ConvBnRelu(backend, pooled, _pspW[k]!, _pspB[k]!, 0);
            pooled.Dispose();
            Tensor up = new(new TensorShape(1, HeadChannels, h, w), DType.F32);
            backend.InterpolateBilinear2D(up, conv, alignCorners: false);
            conv.Dispose();
            pieces[k + 1] = up;
        }
        Tensor cat = new(new TensorShape(1, c + 4 * HeadChannels, h, w), DType.F32);
        backend.Concat(cat, pieces, 1);
        for (int k = 1; k < 5; k++) pieces[k].Dispose();

        Tensor bottleneck = ConvBnRelu(backend, cat, _bottleneckW!, _bottleneckB!, 1);
        cat.Dispose();
        tap?.Invoke("psp_out", bottleneck);
        return bottleneck;
    }

    /// <summary>Folded conv (+BN) followed by ReLU; kernel size comes from the weight shape.</summary>
    private static Tensor ConvBnRelu(IBackend backend, Tensor input, Tensor weight, Tensor bias, int pad)
    {
        Tensor conv = new(new TensorShape(1, weight.Shape[0], input.Shape[2], input.Shape[3]), DType.F32);
        backend.Conv2D(conv, input, weight, bias, 1, 1, pad, pad);
        Tensor relu = new(conv.Shape, DType.F32);
        backend.LeakyRelu(relu, conv, 0f);
        conv.Dispose();
        return relu;
    }

    /// <summary>Channels-first LayerNorm on <c>[1,C,H,W]</c> via a <c>[H·W,C]</c> row view.</summary>
    private static Tensor LayerNormChannelsFirst(IBackend backend, Tensor x, Tensor weight, Tensor bias)
    {
        int c = (int)x.Shape[1];
        long hw = x.Shape[2] * x.Shape[3];
        Tensor rows = new(new TensorShape(hw, c), DType.F32);
        backend.Transpose2D(rows, x.Reshape(new TensorShape(c, hw)), c, (int)hw);
        Tensor normed = new(rows.Shape, DType.F32);
        backend.LayerNorm(normed, rows, weight, bias, LnEps);
        rows.Dispose();
        Tensor output = new(x.Shape, DType.F32);
        backend.Transpose2D(output.Reshape(new TensorShape(c, hw)), normed, (int)hw, c);
        normed.Dispose();
        return output;
    }

    /// <summary>torch <c>AdaptiveAvgPool2d</c> to <paramref name="scale"/>²: cell (i,j) averages input rows
    /// <c>[⌊i·h/s⌋, ⌈(i+1)·h/s⌉)</c>. Host loop — the PSP inputs are tiny (≤ 16×16 at 512² input).</summary>
    private static Tensor AdaptiveAvgPool(Tensor x, int scale)
    {
        int c = (int)x.Shape[1];
        int h = (int)x.Shape[2], w = (int)x.Shape[3];
        Tensor output = new(new TensorShape(1, c, scale, scale), DType.F32);
        float* src = (float*)x.DataPointer;
        float* dst = (float*)output.DataPointer;
        for (int ch = 0; ch < c; ch++)
        {
            float* plane = src + (long)ch * h * w;
            for (int i = 0; i < scale; i++)
            {
                int y0 = i * h / scale;
                int y1 = ((i + 1) * h + scale - 1) / scale;
                for (int j = 0; j < scale; j++)
                {
                    int x0 = j * w / scale;
                    int x1 = ((j + 1) * w + scale - 1) / scale;
                    double sum = 0;
                    for (int y = y0; y < y1; y++)
                        for (int xx = x0; xx < x1; xx++)
                            sum += plane[y * w + xx];
                    dst[((long)ch * scale + i) * scale + j] = (float)(sum / ((y1 - y0) * (x1 - x0)));
                }
            }
        }
        return output;
    }

    /// <summary>Folds the per-channel layer-scale gamma into the second pointwise Linear:
    /// <c>γ ⊙ (Wx + b) = (γ⊙W)x + γ⊙b</c>.</summary>
    private static (Tensor W, Tensor B) FoldLayerScale(Tensor w2, Tensor b2, Tensor gamma)
    {
        int outC = (int)w2.Shape[0];
        long inC = w2.Shape[1];
        Tensor foldedW = new(w2.Shape, DType.F32);
        Tensor foldedB = new(new TensorShape(outC), DType.F32);
        float* pw = (float*)w2.DataPointer;
        float* pfw = (float*)foldedW.DataPointer;
        float* pb = (float*)b2.DataPointer;
        float* pfb = (float*)foldedB.DataPointer;
        float* pg = (float*)gamma.DataPointer;
        for (int o = 0; o < outC; o++)
        {
            for (long i = 0; i < inC; i++)
                pfw[o * inC + i] = pw[o * inC + i] * pg[o];
            pfb[o] = pb[o] * pg[o];
        }
        return (foldedW, foldedB);
    }

    /// <summary>One ConvNeXt block, layer-scale pre-folded into <see cref="Pw2W"/>/<see cref="Pw2B"/>.</summary>
    private sealed class ConvNextBlock
    {
        public Tensor? DwW, DwB, LnW, LnB, Pw1W, Pw1B, Pw2W, Pw2B;

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] all = [DwW, DwB, LnW, LnB, Pw1W, Pw1B, Pw2W, Pw2B];
            foreach (Tensor? t in all) if (t is not null) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor input, int c, int h, int w)
        {
            Tensor dw = new(input.Shape, DType.F32);
            backend.Conv2dDepthwise(dw, input, DwW!, DwB!, 1, 1, 3, 3);

            long hw = (long)h * w;
            Tensor rows = new(new TensorShape(hw, c), DType.F32);
            backend.Transpose2D(rows, dw.Reshape(new TensorShape(c, hw)), c, (int)hw);
            dw.Dispose();
            Tensor normed = new(rows.Shape, DType.F32);
            backend.LayerNorm(normed, rows, LnW!, LnB!, LnEps);
            rows.Dispose();

            Tensor expanded = new(new TensorShape(hw, 4 * c), DType.F32);
            backend.Linear(expanded, normed, Pw1W!, Pw1B!);
            normed.Dispose();
            Tensor act = new(expanded.Shape, DType.F32);
            backend.GeluErf(act, expanded);
            expanded.Dispose();
            Tensor projected = new(new TensorShape(hw, c), DType.F32);
            backend.Linear(projected, act, Pw2W!, Pw2B!);
            act.Dispose();

            Tensor back = new(input.Shape, DType.F32);
            backend.Transpose2D(back.Reshape(new TensorShape(c, hw)), projected, (int)hw, c);
            projected.Dispose();
            Tensor sum = new(input.Shape, DType.F32);
            backend.Add(sum, back, input);
            back.Dispose();
            return sum;
        }
    }
}
