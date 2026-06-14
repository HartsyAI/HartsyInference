using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.Blocks;

/// <summary>YOLO11 decoupled detection head — same box (cv2) branch as YOLOv8 but a lighter
/// depthwise-separable cv3 (class) branch:
/// <list type="bullet">
///   <item><b>cv2 (box)</b>: identical to <see cref="DetectHead"/> — <c>ConvBnSilu(c_in→c2, 3×3) → ConvBnSilu(c2→c2, 3×3) → Conv2D(c2→4·reg_max, 1×1)</c>.</item>
///   <item><b>cv3 (cls)</b>: two depthwise-separable stages then a plain 1×1 projection:
///         <c>[DwConvBnSilu(c_in, 3×3) + ConvBnSilu(c_in→c3, 1×1)] → [DwConvBnSilu(c3, 3×3) + ConvBnSilu(c3→c3, 1×1)] → Conv2D(c3→nc, 1×1)</c>.</item>
/// </list>
/// <para>The decoding (DFL, anchors, sigmoid, output stitch) is identical to v8 — duplicated here
/// rather than abstracted because the per-anchor inner loop is hot and the cv2/cv3 build is the
/// only structural difference. If a third detect-head variant lands later (YOLO12?), promoting
/// the decode tail to a helper makes sense; for now it stays inline.</para></summary>
public sealed unsafe class DetectHeadV11
{
    private readonly int _numClasses;
    private readonly int _regMax;
    private readonly int[] _inChannels;
    private readonly int[] _strides;
    private readonly int _c2; // box-branch hidden width
    private readonly int _c3; // cls-branch hidden width
    private readonly int _numScales;

    // cv2 — same shape as DetectHead.
    private readonly ConvBnSilu[][] _cv2;
    private readonly ConvBnSilu[] _cv2Final;
    // cv3 — depthwise-separable. Two sub-stages, each a (depthwise, pointwise) pair, then a plain 1×1.
    private readonly DwConvBnSilu[][] _cv3Dw;    // _cv3Dw[scale][stage]
    private readonly ConvBnSilu[][] _cv3Pw;      // _cv3Pw[scale][stage]
    private readonly ConvBnSilu[] _cv3Final;

    public int NumClasses => _numClasses;
    public int RegMax => _regMax;
    public IReadOnlyList<int> Strides => _strides;

    public DetectHeadV11(int numClasses, int regMax, int[] inChannels, int[] strides)
    {
        ArgumentNullException.ThrowIfNull(inChannels);
        ArgumentNullException.ThrowIfNull(strides);
        if (inChannels.Length != strides.Length || inChannels.Length == 0)
            throw new ArgumentException("inChannels and strides must be same length and non-empty.");
        if (numClasses <= 0 || regMax <= 0)
            throw new ArgumentOutOfRangeException(nameof(numClasses), "numClasses and regMax must be positive.");

        _numClasses = numClasses;
        _regMax = regMax;
        _inChannels = (int[])inChannels.Clone();
        _strides = (int[])strides.Clone();
        _numScales = inChannels.Length;

        _c2 = Math.Max(16, Math.Max(_inChannels[0] / 4, _regMax * 4));
        _c3 = Math.Max(_inChannels[0], Math.Min(_numClasses, 100));

        _cv2 = new ConvBnSilu[_numScales][];
        _cv2Final = new ConvBnSilu[_numScales];
        _cv3Dw = new DwConvBnSilu[_numScales][];
        _cv3Pw = new ConvBnSilu[_numScales][];
        _cv3Final = new ConvBnSilu[_numScales];

        for (int s = 0; s < _numScales; s++)
        {
            // Box branch — same as v8.
            _cv2[s] = new ConvBnSilu[2];
            _cv2[s][0] = new ConvBnSilu(_c2, 1, 1, 1, 1, useSilu: true);
            _cv2[s][1] = new ConvBnSilu(_c2, 1, 1, 1, 1, useSilu: true);
            _cv2Final[s] = new ConvBnSilu(4 * _regMax, 1, 1, 0, 0, useSilu: false);

            // Class branch — depthwise-separable.
            // Stage 0: depthwise on c_in channels, then pointwise c_in → c3.
            // Stage 1: depthwise on c3 channels, then pointwise c3 → c3.
            // Final: plain Conv2d c3 → nc.
            _cv3Dw[s] = new DwConvBnSilu[2];
            _cv3Pw[s] = new ConvBnSilu[2];
            _cv3Dw[s][0] = new DwConvBnSilu(_inChannels[s], 1, 1, 1, 1, useSilu: true);
            _cv3Pw[s][0] = new ConvBnSilu(_c3, 1, 1, 0, 0, useSilu: true);
            _cv3Dw[s][1] = new DwConvBnSilu(_c3, 1, 1, 1, 1, useSilu: true);
            _cv3Pw[s][1] = new ConvBnSilu(_c3, 1, 1, 0, 0, useSilu: true);
            _cv3Final[s] = new ConvBnSilu(_numClasses, 1, 1, 0, 0, useSilu: false);
        }
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        for (int s = 0; s < _numScales; s++)
        {
            _cv2[s][0].LoadWeights(weights, $"{prefix}.cv2.{s}.0.conv");
            _cv2[s][1].LoadWeights(weights, $"{prefix}.cv2.{s}.1.conv");
            _cv2Final[s].LoadWeights(weights, $"{prefix}.cv2.{s}.2");

            // Class branch uses nested Sequentials: cv3.{s}.{stage}.{0=dw, 1=pw}.conv.* + cv3.{s}.2.{weight,bias}
            _cv3Dw[s][0].LoadWeights(weights, $"{prefix}.cv3.{s}.0.0.conv");
            _cv3Pw[s][0].LoadWeights(weights, $"{prefix}.cv3.{s}.0.1.conv");
            _cv3Dw[s][1].LoadWeights(weights, $"{prefix}.cv3.{s}.1.0.conv");
            _cv3Pw[s][1].LoadWeights(weights, $"{prefix}.cv3.{s}.1.1.conv");
            _cv3Final[s].LoadWeights(weights, $"{prefix}.cv3.{s}.2");
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int s = 0; s < _numScales; s++)
        {
            foreach (Tensor t in _cv2[s][0].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv2[s][1].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv2Final[s].EnumerateWeights()) yield return t;

            foreach (Tensor t in _cv3Dw[s][0].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv3Pw[s][0].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv3Dw[s][1].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv3Pw[s][1].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv3Final[s].EnumerateWeights()) yield return t;
        }
    }

    /// <summary>Forward + decode. Returns <c>[B, 4 + numClasses, totalAnchors]</c> with xywh boxes
    /// in input-pixel coords and sigmoid'd class probs — same contract as <see cref="DetectHead.Forward"/>.</summary>
    public Tensor Forward(IBackend backend, IReadOnlyList<Tensor> features)
    {
        ArgumentNullException.ThrowIfNull(features);
        if (features.Count != _numScales)
            throw new ArgumentException($"Detect head expects {_numScales} feature tensors; got {features.Count}.");

        int batch = (int)features[0].Shape[0];

        Tensor[] boxLogits = new Tensor[_numScales];
        Tensor[] clsLogits = new Tensor[_numScales];
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

            // Class branch — depthwise → pointwise → depthwise → pointwise → final 1×1.
            Tensor c1 = _cv3Dw[s][0].Forward(backend, feat);
            Tensor c2 = _cv3Pw[s][0].Forward(backend, c1);
            c1.Dispose();
            Tensor c3 = _cv3Dw[s][1].Forward(backend, c2);
            c2.Dispose();
            Tensor c4 = _cv3Pw[s][1].Forward(backend, c3);
            c3.Dispose();
            clsLogits[s] = _cv3Final[s].Forward(backend, c4);
            c4.Dispose();
        }

        Tensor output = new Tensor(new TensorShape(batch, 4 + _numClasses, totalAnchors), DType.F32);
        float* outPtr = (float*)output.DataPointer;

        // Scratch buffers reused across all anchors (CA2014: no stackalloc in loops).
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
            int boxChStride = 4 * _regMax * numAnchorsScale;
            int clsChStride = _numClasses * numAnchorsScale;

            for (int bIdx = 0; bIdx < batch; bIdx++)
            {
                float* boxB = boxPtr + bIdx * boxChStride;
                float* clsB = clsPtr + bIdx * clsChStride;

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
                        long outBatchStride = (long)(4 + _numClasses) * totalAnchors;
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
                    }
                }
            }
            anchorOffset += numAnchorsScale;
        }

        for (int s = 0; s < _numScales; s++)
        {
            boxLogits[s].Dispose();
            clsLogits[s].Dispose();
        }
        return output;
    }
}
