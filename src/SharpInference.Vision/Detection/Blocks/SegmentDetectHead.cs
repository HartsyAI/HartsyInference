using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Vision.Detection.Blocks;

/// <summary>YOLOv8-seg detection head — extends the standard <see cref="DetectHead"/> with a
/// per-scale <b>cv4 mask-coefficient branch</b> (3×3 → 3×3 → 1×1, output <c>nm</c> channels) and a
/// <see cref="Proto"/> mask prototype generator fed from the P3 feature map.
/// <para>Outputs:</para>
/// <list type="bullet">
///   <item><b>Box logits</b> + <b>class logits</b> + <b>mask coefficients</b>: stacked into a
///         single <c>[B, 4 + nc + nm, totalAnchors]</c> tensor (box xywh in pixel coords, class
///         probs after sigmoid, mask coefs untouched).</item>
///   <item><b>Mask prototypes</b>: <c>[B, nm, protoH, protoW]</c> where (protoH, protoW) is
///         <c>2 × P3</c> spatial dims. Returned separately — the post-processor combines them
///         with the per-detection coefficients via <see cref="MaskAssembly.AssembleMasks"/>.</item>
/// </list>
/// <para>This class is structurally aligned with <see cref="DetectHead"/> — same cv2/cv3 box/cls
/// branches and the same DFL + anchor + sigmoid decoding. The only additions are the cv4 branch
/// and the Proto module.</para></summary>
public sealed unsafe class SegmentDetectHead
{
    private readonly int _numClasses;
    private readonly int _numMasks;
    private readonly int _regMax;
    private readonly int[] _inChannels;
    private readonly int[] _strides;
    private readonly int _c2;       // box-branch hidden
    private readonly int _c3;       // cls-branch hidden
    private readonly int _c4;       // mask-coef-branch hidden
    private readonly int _numScales;

    // Box (cv2) — same as DetectHead.
    private readonly ConvBnSilu[][] _cv2;
    private readonly ConvBnSilu[] _cv2Final;
    // Class (cv3) — same as DetectHead.
    private readonly ConvBnSilu[][] _cv3;
    private readonly ConvBnSilu[] _cv3Final;
    // Mask coefficient (cv4) — new for segmentation.
    private readonly ConvBnSilu[][] _cv4;
    private readonly ConvBnSilu[] _cv4Final;
    // Proto module.
    private readonly Proto _proto;

    public int NumClasses => _numClasses;
    public int NumMasks => _numMasks;
    public int RegMax => _regMax;
    public IReadOnlyList<int> Strides => _strides;
    public Proto ProtoModule => _proto;

    public SegmentDetectHead(int numClasses, int numMasks, int regMax, int[] inChannels, int[] strides, int protoInternalChannels)
    {
        ArgumentNullException.ThrowIfNull(inChannels);
        ArgumentNullException.ThrowIfNull(strides);
        if (inChannels.Length != strides.Length || inChannels.Length == 0)
            throw new ArgumentException("inChannels and strides must be same length and non-empty.");

        _numClasses = numClasses;
        _numMasks = numMasks;
        _regMax = regMax;
        _inChannels = (int[])inChannels.Clone();
        _strides = (int[])strides.Clone();
        _numScales = inChannels.Length;

        _c2 = Math.Max(16, Math.Max(_inChannels[0] / 4, _regMax * 4));
        _c3 = Math.Max(_inChannels[0], Math.Min(_numClasses, 100));
        // Ultralytics: c4 = max(ch[0] // 4, nm)
        _c4 = Math.Max(_inChannels[0] / 4, _numMasks);

        _cv2 = new ConvBnSilu[_numScales][];
        _cv2Final = new ConvBnSilu[_numScales];
        _cv3 = new ConvBnSilu[_numScales][];
        _cv3Final = new ConvBnSilu[_numScales];
        _cv4 = new ConvBnSilu[_numScales][];
        _cv4Final = new ConvBnSilu[_numScales];

        for (int s = 0; s < _numScales; s++)
        {
            _cv2[s] = new ConvBnSilu[2];
            _cv2[s][0] = new ConvBnSilu(_c2, 1, 1, 1, 1, useSilu: true);
            _cv2[s][1] = new ConvBnSilu(_c2, 1, 1, 1, 1, useSilu: true);
            _cv2Final[s] = new ConvBnSilu(4 * _regMax, 1, 1, 0, 0, useSilu: false);

            _cv3[s] = new ConvBnSilu[2];
            _cv3[s][0] = new ConvBnSilu(_c3, 1, 1, 1, 1, useSilu: true);
            _cv3[s][1] = new ConvBnSilu(_c3, 1, 1, 1, 1, useSilu: true);
            _cv3Final[s] = new ConvBnSilu(_numClasses, 1, 1, 0, 0, useSilu: false);

            _cv4[s] = new ConvBnSilu[2];
            _cv4[s][0] = new ConvBnSilu(_c4, 1, 1, 1, 1, useSilu: true);
            _cv4[s][1] = new ConvBnSilu(_c4, 1, 1, 1, 1, useSilu: true);
            _cv4Final[s] = new ConvBnSilu(_numMasks, 1, 1, 0, 0, useSilu: false);
        }

        _proto = new Proto(_inChannels[0], protoInternalChannels, _numMasks);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        for (int s = 0; s < _numScales; s++)
        {
            _cv2[s][0].LoadWeights(weights, $"{prefix}.cv2.{s}.0.conv");
            _cv2[s][1].LoadWeights(weights, $"{prefix}.cv2.{s}.1.conv");
            _cv2Final[s].LoadWeights(weights, $"{prefix}.cv2.{s}.2");

            _cv3[s][0].LoadWeights(weights, $"{prefix}.cv3.{s}.0.conv");
            _cv3[s][1].LoadWeights(weights, $"{prefix}.cv3.{s}.1.conv");
            _cv3Final[s].LoadWeights(weights, $"{prefix}.cv3.{s}.2");

            _cv4[s][0].LoadWeights(weights, $"{prefix}.cv4.{s}.0.conv");
            _cv4[s][1].LoadWeights(weights, $"{prefix}.cv4.{s}.1.conv");
            _cv4Final[s].LoadWeights(weights, $"{prefix}.cv4.{s}.2");
        }
        _proto.LoadWeights(weights, $"{prefix}.proto");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int s = 0; s < _numScales; s++)
        {
            foreach (Tensor t in _cv2[s][0].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv2[s][1].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv2Final[s].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv3[s][0].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv3[s][1].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv3Final[s].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv4[s][0].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv4[s][1].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv4Final[s].EnumerateWeights()) yield return t;
        }
        foreach (Tensor t in _proto.EnumerateWeights()) yield return t;
    }

    /// <summary>Runs the seg head. Returns <c>(detections [B, 4+nc+nm, totalAnchors], prototypes [B, nm, protoH, protoW])</c>.</summary>
    public (Tensor detections, Tensor prototypes) Forward(IBackend backend, IReadOnlyList<Tensor> features)
    {
        ArgumentNullException.ThrowIfNull(features);
        if (features.Count != _numScales)
            throw new ArgumentException($"Segment head expects {_numScales} feature tensors; got {features.Count}.");

        int batch = (int)features[0].Shape[0];

        // Proto on P3 features.
        Tensor protos = _proto.Forward(backend, features[0]);

        // Per-scale logits.
        Tensor[] boxLogits = new Tensor[_numScales];
        Tensor[] clsLogits = new Tensor[_numScales];
        Tensor[] maskCoefs = new Tensor[_numScales];
        int[] heights = new int[_numScales];
        int[] widths = new int[_numScales];
        int totalAnchors = 0;

        for (int s = 0; s < _numScales; s++)
        {
            Tensor feat = features[s];
            heights[s] = (int)feat.Shape[2];
            widths[s] = (int)feat.Shape[3];
            totalAnchors += heights[s] * widths[s];

            // Box branch.
            Tensor b1 = _cv2[s][0].Forward(backend, feat);
            Tensor b2 = _cv2[s][1].Forward(backend, b1);
            b1.Dispose();
            boxLogits[s] = _cv2Final[s].Forward(backend, b2);
            b2.Dispose();

            // Class branch.
            Tensor c1 = _cv3[s][0].Forward(backend, feat);
            Tensor c2 = _cv3[s][1].Forward(backend, c1);
            c1.Dispose();
            clsLogits[s] = _cv3Final[s].Forward(backend, c2);
            c2.Dispose();

            // Mask-coefficient branch.
            Tensor m1 = _cv4[s][0].Forward(backend, feat);
            Tensor m2 = _cv4[s][1].Forward(backend, m1);
            m1.Dispose();
            maskCoefs[s] = _cv4Final[s].Forward(backend, m2);
            m2.Dispose();
        }

        int outChannels = 4 + _numClasses + _numMasks;
        Tensor detections = new Tensor(new TensorShape(batch, outChannels, totalAnchors), DType.F32);
        float* outPtr = (float*)detections.DataPointer;

        Span<float> distances = stackalloc float[4];
        Span<float> probs = stackalloc float[_regMax];

        int anchorOffset = 0;
        for (int s = 0; s < _numScales; s++)
        {
            int h = heights[s];
            int w = widths[s];
            int numAnchorsScale = h * w;
            float stride = _strides[s];

            float* boxPtr = (float*)boxLogits[s].DataPointer;
            float* clsPtr = (float*)clsLogits[s].DataPointer;
            float* maskPtr = (float*)maskCoefs[s].DataPointer;
            int boxChStride = 4 * _regMax * numAnchorsScale;
            int clsChStride = _numClasses * numAnchorsScale;
            int maskChStride = _numMasks * numAnchorsScale;

            for (int bIdx = 0; bIdx < batch; bIdx++)
            {
                float* boxB = boxPtr + bIdx * boxChStride;
                float* clsB = clsPtr + bIdx * clsChStride;
                float* maskB = maskPtr + bIdx * maskChStride;

                for (int gy = 0; gy < h; gy++)
                {
                    for (int gx = 0; gx < w; gx++)
                    {
                        int anchorIdx = gy * w + gx;
                        for (int dim = 0; dim < 4; dim++)
                        {
                            float maxLogit = float.NegativeInfinity;
                            for (int bin = 0; bin < _regMax; bin++)
                            {
                                int ch = dim * _regMax + bin;
                                float v = boxB[ch * numAnchorsScale + anchorIdx];
                                if (v > maxLogit) maxLogit = v;
                            }
                            float sumExp = 0f;
                            for (int bin = 0; bin < _regMax; bin++)
                            {
                                int ch = dim * _regMax + bin;
                                float e = MathF.Exp(boxB[ch * numAnchorsScale + anchorIdx] - maxLogit);
                                probs[bin] = e;
                                sumExp += e;
                            }
                            float inv = sumExp > 0f ? 1f / sumExp : 0f;
                            float dist = 0f;
                            for (int bin = 0; bin < _regMax; bin++)
                                dist += probs[bin] * inv * bin;
                            distances[dim] = dist;
                        }

                        float anchorX = gx + 0.5f;
                        float anchorY = gy + 0.5f;
                        float l = distances[0], t = distances[1], r = distances[2], btm = distances[3];
                        float cx = (anchorX - l + anchorX + r) * 0.5f * stride;
                        float cy = (anchorY - t + anchorY + btm) * 0.5f * stride;
                        float bw = (l + r) * stride;
                        float bh = (t + btm) * stride;

                        long outAnchor = anchorOffset + anchorIdx;
                        long outBatchStride = (long)outChannels * totalAnchors;
                        long outBase = bIdx * outBatchStride;
                        outPtr[outBase + 0L * totalAnchors + outAnchor] = cx;
                        outPtr[outBase + 1L * totalAnchors + outAnchor] = cy;
                        outPtr[outBase + 2L * totalAnchors + outAnchor] = bw;
                        outPtr[outBase + 3L * totalAnchors + outAnchor] = bh;
                        for (int k = 0; k < _numClasses; k++)
                        {
                            float logit = clsB[k * numAnchorsScale + anchorIdx];
                            float prob = 1f / (1f + MathF.Exp(-logit));
                            outPtr[outBase + (long)(4 + k) * totalAnchors + outAnchor] = prob;
                        }
                        // Mask coefficients are stored RAW (no sigmoid) — Ultralytics applies
                        // sigmoid only after combining with prototypes in MaskAssembly.
                        for (int k = 0; k < _numMasks; k++)
                        {
                            float coef = maskB[k * numAnchorsScale + anchorIdx];
                            outPtr[outBase + (long)(4 + _numClasses + k) * totalAnchors + outAnchor] = coef;
                        }
                    }
                }
            }
            anchorOffset += numAnchorsScale;
        }

        for (int s = 0; s < _numScales; s++)
        {
            boxLogits[s].Dispose();
            clsLogits[s].Dispose();
            maskCoefs[s].Dispose();
        }
        return (detections, protos);
    }
}
