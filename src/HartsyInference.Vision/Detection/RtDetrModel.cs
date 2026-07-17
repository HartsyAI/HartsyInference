using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>End-to-end RT-DETR model: ResNet-vd backbone → encoder input projection → hybrid encoder
/// (AIFI + CCFM) → decoder input projection → two-stage query selection → deformable-attention decoder
/// → class + box heads. NMS-free; emits exactly <c>NumQueries</c> candidate detections.
/// <para>Input is <c>[1, 3, H, W]</c> (RT-DETR uses 640×640). Output is class logits
/// <c>[1, NumQueries, NumClasses]</c> (pre-sigmoid) and boxes <c>[1, NumQueries, 4]</c> as cxcywh
/// normalized to [0, 1].</para></summary>
public sealed class RtDetrModel
{
    private const string Root = "model";

    private readonly RtDetrConfig _config;
    private readonly RtDetrBackbone _backbone;
    private readonly RtDetrConv[] _encoderInputProj;
    private readonly RtDetrHybridEncoder _encoder;
    private readonly RtDetrConv[] _decoderInputProj;
    private readonly RtDetrLinear _encOutputLinear;
    private readonly RtDetrLayerNorm _encOutputNorm;
    private readonly RtDetrLinear _encScoreHead;
    private readonly RtDetrMlp _encBboxHead;
    private readonly RtDetrDecoder _decoder;

    /// <summary>The config this model was constructed with.</summary>
    public RtDetrConfig Config => _config;

    /// <summary>Number of detection classes.</summary>
    public int NumClasses => _config.NumClasses;

    /// <summary>Number of object queries (= max detections emitted per image).</summary>
    public int NumQueries => _config.NumQueries;

    /// <summary>The deformable-attention decoder — exposed for structural tests + diagnostics.</summary>
    public RtDetrDecoder Decoder => _decoder;

    /// <summary>Builds the model from a config (validated here).</summary>
    public RtDetrModel(RtDetrConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        _config = config;
        _backbone = new RtDetrBackbone(config);
        _encoderInputProj = [Proj1x1(), Proj1x1(), Proj1x1()];
        _encoder = new RtDetrHybridEncoder(config);
        _decoderInputProj = [Proj1x1(), Proj1x1(), Proj1x1()];
        _encOutputLinear = new RtDetrLinear(config.HiddenDim);
        _encOutputNorm = new RtDetrLayerNorm(config.HiddenDim, config.LayerNormEps);
        _encScoreHead = new RtDetrLinear(config.NumClasses);
        _encBboxHead = new RtDetrMlp([config.HiddenDim, config.HiddenDim, 4]);
        _decoder = new RtDetrDecoder(config);

        static RtDetrConv Proj1x1() => new RtDetrConv(1, 1, 0, 0, RtDetrActivation.None);
    }

    /// <summary>Loads all weights (from a converted, BN-folded dict). Keys use the HF checkpoint
    /// namespace under <c>model.*</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _backbone.LoadWeights(weights, $"{Root}.backbone.model");
        for (int i = 0; i < 3; i++)
        {
            _encoderInputProj[i].LoadWeights(weights, $"{Root}.encoder_input_proj.{i}.0");
            _decoderInputProj[i].LoadWeights(weights, $"{Root}.decoder_input_proj.{i}.0");
        }
        _encoder.LoadWeights(weights, $"{Root}.encoder");
        _encOutputLinear.LoadWeights(weights, $"{Root}.enc_output.0");
        _encOutputNorm.LoadWeights(weights, $"{Root}.enc_output.1");
        _encScoreHead.LoadWeights(weights, $"{Root}.enc_score_head");
        _encBboxHead.LoadWeights(weights, $"{Root}.enc_bbox_head");
        _decoder.LoadWeights(weights, $"{Root}.decoder");
    }

    /// <summary>Yields every weight tensor for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
        for (int i = 0; i < 3; i++)
        {
            foreach (Tensor t in _encoderInputProj[i].EnumerateWeights()) yield return t;
            foreach (Tensor t in _decoderInputProj[i].EnumerateWeights()) yield return t;
        }
        foreach (Tensor t in _encoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _encOutputLinear.EnumerateWeights()) yield return t;
        foreach (Tensor t in _encOutputNorm.EnumerateWeights()) yield return t;
        foreach (Tensor t in _encScoreHead.EnumerateWeights()) yield return t;
        foreach (Tensor t in _encBboxHead.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decoder.EnumerateWeights()) yield return t;
    }

    /// <summary>Runs the full forward pass. Returns class logits <c>[1, NumQueries, NumClasses]</c>
    /// (pre-sigmoid) and boxes <c>[1, NumQueries, 4]</c> (cxcywh in [0,1]). Caller owns both.</summary>
    public Tensor Forward(IBackend backend, Tensor pixelValues, out Tensor boxes)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(pixelValues);
        if (pixelValues.Shape.Rank != 4 || pixelValues.Shape[1] != 3)
            throw new ArgumentException($"RT-DETR input must be [1, 3, H, W]; got {pixelValues.Shape}.", nameof(pixelValues));

        (Tensor c3, Tensor c4, Tensor c5) = _backbone.ForwardFeatures(backend, pixelValues);
        Tensor f0 = _encoderInputProj[0].Forward(backend, c3);
        Tensor f1 = _encoderInputProj[1].Forward(backend, c4);
        Tensor f2 = _encoderInputProj[2].Forward(backend, c5);
        c3.Dispose(); c4.Dispose(); c5.Dispose();

        (Tensor pan0, Tensor pan1, Tensor pan2) = _encoder.Forward(backend, f0, f1, f2);
        f0.Dispose(); f1.Dispose(); f2.Dispose();

        Tensor s0 = _decoderInputProj[0].Forward(backend, pan0);
        Tensor s1 = _decoderInputProj[1].Forward(backend, pan1);
        Tensor s2 = _decoderInputProj[2].Forward(backend, pan2);
        pan0.Dispose(); pan1.Dispose(); pan2.Dispose();

        (int H, int W)[] shapes =
        [
            ((int)s0.Shape[2], (int)s0.Shape[3]),
            ((int)s1.Shape[2], (int)s1.Shape[3]),
            ((int)s2.Shape[2], (int)s2.Shape[3]),
        ];
        int[] levelStart = new int[3];
        int seq = 0;
        for (int l = 0; l < 3; l++) { levelStart[l] = seq; seq += shapes[l].H * shapes[l].W; }

        Tensor sourceFlatten = FlattenLevels(s0, s1, s2, seq, _config.HiddenDim);
        s0.Dispose(); s1.Dispose(); s2.Dispose();

        (Tensor anchors, bool[] validMask) = GenerateAnchors(shapes, _config.AnchorGridSize);

        Tensor memory = ApplyValidMask(sourceFlatten, validMask, _config.HiddenDim);
        Tensor outMemLinear = _encOutputLinear.Forward(backend, memory);
        memory.Dispose();
        Tensor outputMemory = _encOutputNorm.Forward(backend, outMemLinear);
        outMemLinear.Dispose();

        Tensor encClass = _encScoreHead.Forward(backend, outputMemory);   // [1, S, C]
        Tensor encCoordDelta = _encBboxHead.Forward(backend, outputMemory); // [1, S, 4]
        Tensor encCoord = AddAnchors(encCoordDelta, anchors);
        encCoordDelta.Dispose(); anchors.Dispose();

        int[] topk = TopKByMaxClass(encClass.AsSpan<float>(), seq, _config.NumClasses, _config.NumQueries);
        Tensor referenceUnact = Gather(encCoord.AsSpan<float>(), topk, 4);
        Tensor target = Gather(outputMemory.AsSpan<float>(), topk, _config.HiddenDim);
        encClass.Dispose(); encCoord.Dispose(); outputMemory.Dispose();

        Tensor logits = _decoder.Forward(backend, target, referenceUnact, sourceFlatten, shapes, levelStart, out boxes);
        target.Dispose(); referenceUnact.Dispose(); sourceFlatten.Dispose();
        return logits;
    }

    private static unsafe Tensor FlattenLevels(Tensor s0, Tensor s1, Tensor s2, int seq, int d)
    {
        Tensor outp = new Tensor(new TensorShape(1, seq, d), DType.F32);
        float* dst = (float*)outp.DataPointer;
        int offset = 0;
        foreach (Tensor s in new[] { s0, s1, s2 })
        {
            int h = (int)s.Shape[2], w = (int)s.Shape[3], hw = h * w;
            float* src = (float*)s.DataPointer;
            for (int p = 0; p < hw; p++)
                for (int ci = 0; ci < d; ci++)
                    dst[(long)(offset + p) * d + ci] = src[(long)ci * hw + p];
            offset += hw;
        }
        return outp;
    }

    /// <summary>Generates the two-stage anchors: per token <c>[(x+0.5)/W, (y+0.5)/H, grid·2^l, grid·2^l]</c>,
    /// logit-transformed (or set to finfo.max where out of the valid [eps, 1-eps] range). Returns the
    /// <c>[1, S, 4]</c> anchor tensor and the per-token valid mask.</summary>
    private static unsafe (Tensor Anchors, bool[] Valid) GenerateAnchors((int H, int W)[] shapes, float gridSize)
    {
        int seq = 0;
        foreach ((int H, int W) in shapes) seq += H * W;
        Tensor anchors = new Tensor(new TensorShape(1, seq, 4), DType.F32);
        float* dst = (float*)anchors.DataPointer;
        bool[] valid = new bool[seq];
        const float eps = 1e-2f;
        float fmax = float.MaxValue;
        int token = 0;
        for (int l = 0; l < shapes.Length; l++)
        {
            int h = shapes[l].H, w = shapes[l].W;
            float wh = gridSize * MathF.Pow(2f, l);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float cx = (x + 0.5f) / w;
                    float cy = (y + 0.5f) / h;
                    float a0 = cx, a1 = cy, a2 = wh, a3 = wh;
                    bool v = a0 > eps && a0 < 1 - eps && a1 > eps && a1 < 1 - eps
                          && a2 > eps && a2 < 1 - eps && a3 > eps && a3 < 1 - eps;
                    valid[token] = v;
                    float* row = dst + (long)token * 4;
                    if (v)
                    {
                        row[0] = MathF.Log(a0 / (1 - a0));
                        row[1] = MathF.Log(a1 / (1 - a1));
                        row[2] = MathF.Log(a2 / (1 - a2));
                        row[3] = MathF.Log(a3 / (1 - a3));
                    }
                    else
                    {
                        row[0] = fmax; row[1] = fmax; row[2] = fmax; row[3] = fmax;
                    }
                    token++;
                }
        }
        return (anchors, valid);
    }

    private static unsafe Tensor ApplyValidMask(Tensor sourceFlatten, bool[] valid, int d)
    {
        Tensor memory = new Tensor(sourceFlatten.Shape, DType.F32);
        float* src = (float*)sourceFlatten.DataPointer, dst = (float*)memory.DataPointer;
        for (int t = 0; t < valid.Length; t++)
        {
            long baseIdx = (long)t * d;
            if (valid[t])
                for (int c = 0; c < d; c++) dst[baseIdx + c] = src[baseIdx + c];
            else
                for (int c = 0; c < d; c++) dst[baseIdx + c] = 0f;
        }
        return memory;
    }

    private static Tensor AddAnchors(Tensor coordDelta, Tensor anchors)
    {
        Tensor outp = new Tensor(coordDelta.Shape, DType.F32);
        Span<float> o = outp.AsSpan<float>();
        ReadOnlySpan<float> a = coordDelta.AsSpan<float>(), b = anchors.AsSpan<float>();
        for (int i = 0; i < o.Length; i++)
            o[i] = a[i] + b[i];
        return outp;
    }

    /// <summary>Top-<paramref name="k"/> token indices by max-over-classes score (descending; ties by
    /// lower index), matching <c>torch.topk(scores.max(-1).values, k)</c>.</summary>
    private static int[] TopKByMaxClass(ReadOnlySpan<float> classLogits, int seq, int numClasses, int k)
    {
        float[] maxScore = new float[seq];
        for (int t = 0; t < seq; t++)
        {
            float m = float.NegativeInfinity;
            int baseIdx = t * numClasses;
            for (int c = 0; c < numClasses; c++)
                if (classLogits[baseIdx + c] > m) m = classLogits[baseIdx + c];
            maxScore[t] = m;
        }
        int[] idx = new int[seq];
        for (int i = 0; i < seq; i++) idx[i] = i;
        Array.Sort(idx, (a, b) =>
        {
            int cmp = maxScore[b].CompareTo(maxScore[a]);
            return cmp != 0 ? cmp : a.CompareTo(b);
        });
        int[] top = new int[k];
        Array.Copy(idx, top, k);
        return top;
    }

    private static unsafe Tensor Gather(ReadOnlySpan<float> source, int[] indices, int d)
    {
        Tensor outp = new Tensor(new TensorShape(1, indices.Length, d), DType.F32);
        Span<float> o = outp.AsSpan<float>();
        for (int i = 0; i < indices.Length; i++)
        {
            int src = indices[i] * d;
            int dst = i * d;
            for (int c = 0; c < d; c++)
                o[dst + c] = source[src + c];
        }
        return outp;
    }
}
