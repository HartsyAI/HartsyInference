using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection.Blocks;

namespace HartsyInference.Vision.FaceDetection;

/// <summary>YOLOv8-Face detect head — the stock YOLOv8 <see cref="DetectHead"/> (box <c>cv2</c> + 1-class <c>cv3</c>,
/// decoded to <c>[B, 4+1, A]</c>) plus a landmark branch (<c>cv4</c>) decoded to <c>[B, 5·ndim, A]</c>. This is the
/// only structurally-new piece of YOLOv8-Face: everything else is the shared detection stack.
///
/// <para>The <c>cv4</c> branch mirrors Ultralytics' pose keypoint head — per detect scale
/// <c>cv4 = Conv(c_in→c4, 3×3) → Conv(c4→c4, 3×3) → Conv2d(c4→5·ndim, 1×1)</c> with <c>c4 = max(c_in[0]/4, 5·ndim)</c>,
/// and the raw outputs decode per anchor as <c>x = (2·rx + gx)·stride</c>, <c>y = (2·ry + gy)·stride</c>,
/// <c>v = σ(rv)</c> (grid-cell offset 0.5, matching <c>kpts_decode</c>). The box/class decode is delegated to the
/// composed <see cref="DetectHead"/>, so DFL/anchor/sigmoid logic is not duplicated; this class owns only the
/// landmark branch and its grid decode.</para>
///
/// <para>Reuse note: YOLOv8-Face keeps the v8 <see cref="DetectHead"/> (dense <c>cv3</c>), whereas the existing
/// <see cref="PoseHeadV11"/> composes the v11 head (depthwise-separable <c>cv3</c>). The landmark branch itself is
/// identical in shape to the pose keypoint branch — only the number of points (5 vs 17) differs.</para></summary>
public sealed unsafe class FaceDetectHead
{
    private readonly DetectHead _detect;
    private readonly int _numPoints;   // landmark points per face (5)
    private readonly int _kptDims;     // values per point (2 = x,y  |  3 = x,y,visibility)
    private readonly int _nkFlat;      // _numPoints * _kptDims
    private readonly int _c4;          // landmark-branch hidden width
    private readonly int[] _inChannels;
    private readonly int[] _strides;
    private readonly int _numScales;

    private readonly ConvBnSilu[][] _cv4;    // _cv4[scale][0..1] — the two 3×3 stages
    private readonly ConvBnSilu[] _cv4Final; // _cv4Final[scale] — the 1×1 projection (no BN/SiLU)

    /// <summary>Detection classes (1 for a face detector).</summary>
    public int NumClasses => _detect.NumClasses;

    /// <summary>Landmark points per face (5).</summary>
    public int NumPoints => _numPoints;

    /// <summary>Values per landmark point (2 or 3).</summary>
    public int KptDims => _kptDims;

    /// <summary>Per-detect-scale strides — <c>[8, 16, 32]</c>.</summary>
    public IReadOnlyList<int> Strides => _strides;

    public FaceDetectHead(int numClasses, int regMax, int numPoints, int kptDims, int[] inChannels, int[] strides)
    {
        ArgumentNullException.ThrowIfNull(inChannels);
        ArgumentNullException.ThrowIfNull(strides);
        if (numPoints <= 0 || kptDims < 2 || kptDims > 3)
            throw new ArgumentOutOfRangeException(nameof(kptDims), "numPoints must be positive and kptDims must be 2 or 3.");

        _detect = new DetectHead(numClasses, regMax, inChannels, strides);
        _numPoints = numPoints;
        _kptDims = kptDims;
        _nkFlat = numPoints * kptDims;
        _inChannels = (int[])inChannels.Clone();
        _strides = (int[])strides.Clone();
        _numScales = inChannels.Length;
        _c4 = Math.Max(_inChannels[0] / 4, _nkFlat);

        _cv4 = new ConvBnSilu[_numScales][];
        _cv4Final = new ConvBnSilu[_numScales];
        for (int s = 0; s < _numScales; s++)
        {
            _cv4[s] = new ConvBnSilu[2];
            _cv4[s][0] = new ConvBnSilu(_c4, 1, 1, 1, 1, useSilu: true);   // 3×3, pad 1 (kernel size read from weights)
            _cv4[s][1] = new ConvBnSilu(_c4, 1, 1, 1, 1, useSilu: true);   // 3×3, pad 1
            _cv4Final[s] = new ConvBnSilu(_nkFlat, 1, 1, 0, 0, useSilu: false);   // 1×1 plain Conv2d, no BN/SiLU
        }
    }

    /// <summary>Loads head weights. Ultralytics face keys follow <c>{prefix}.cv2/cv3/cv4.{scale}.{stage}...</c> where
    /// <c>prefix</c> is the detect layer (<c>model.22</c> for a v8 backbone).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _detect.LoadWeights(weights, prefix);
        for (int s = 0; s < _numScales; s++)
        {
            _cv4[s][0].LoadWeights(weights, $"{prefix}.cv4.{s}.0.conv");
            _cv4[s][1].LoadWeights(weights, $"{prefix}.cv4.{s}.1.conv");
            _cv4Final[s].LoadWeights(weights, $"{prefix}.cv4.{s}.2");
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _detect.EnumerateWeights()) yield return t;
        for (int s = 0; s < _numScales; s++)
        {
            foreach (Tensor t in _cv4[s][0].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv4[s][1].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv4Final[s].EnumerateWeights()) yield return t;
        }
    }

    /// <summary>Returns the box/class detections <c>[B, 4+nc, A]</c> (input-pixel xywh + sigmoid probs) and the decoded
    /// landmarks <c>[B, 5·ndim, A]</c> (x, y in input-pixel coords, visibility sigmoid'd when ndim = 3), over the same
    /// concatenated P3/P4/P5 anchor order. Caller owns disposal of both.</summary>
    public (Tensor detections, Tensor landmarks) Forward(IBackend backend, IReadOnlyList<Tensor> features)
    {
        ArgumentNullException.ThrowIfNull(features);
        if (features.Count != _numScales)
            throw new ArgumentException($"Face head expects {_numScales} feature tensors; got {features.Count}.");

        Tensor detections = _detect.Forward(backend, features);
        int batch = (int)features[0].Shape[0];

        Tensor[] lmLogits = new Tensor[_numScales];
        int[] heights = new int[_numScales];
        int[] widths = new int[_numScales];
        int totalAnchors = 0;
        for (int s = 0; s < _numScales; s++)
        {
            Tensor feat = features[s];
            heights[s] = (int)feat.Shape[2];
            widths[s] = (int)feat.Shape[3];
            totalAnchors += heights[s] * widths[s];
            Tensor l1 = _cv4[s][0].Forward(backend, feat);
            Tensor l2 = _cv4[s][1].Forward(backend, l1);
            l1.Dispose();
            lmLogits[s] = _cv4Final[s].Forward(backend, l2);
            l2.Dispose();
        }

        Tensor landmarks = new Tensor(new TensorShape(batch, _nkFlat, totalAnchors), DType.F32);
        float* outPtr = (float*)landmarks.DataPointer;
        long outBatchStride = (long)_nkFlat * totalAnchors;

        int anchorOffset = 0;
        for (int s = 0; s < _numScales; s++)
        {
            int h = heights[s], w = widths[s];
            int numAnchorsScale = h * w;
            float stride = _strides[s];
            float* lmPtr = (float*)lmLogits[s].DataPointer;
            int chStride = _nkFlat * numAnchorsScale;

            for (int bIdx = 0; bIdx < batch; bIdx++)
            {
                float* lmB = lmPtr + bIdx * chStride;
                for (int gy = 0; gy < h; gy++)
                {
                    for (int gx = 0; gx < w; gx++)
                    {
                        int anchorIdx = gy * w + gx;
                        long outAnchor = anchorOffset + anchorIdx;
                        long outBase = bIdx * outBatchStride;
                        for (int k = 0; k < _numPoints; k++)
                        {
                            int xCh = k * _kptDims + 0;
                            int yCh = k * _kptDims + 1;
                            float rx = lmB[(long)xCh * numAnchorsScale + anchorIdx];
                            float ry = lmB[(long)yCh * numAnchorsScale + anchorIdx];
                            outPtr[outBase + (long)xCh * totalAnchors + outAnchor] = (rx * 2f + gx) * stride;
                            outPtr[outBase + (long)yCh * totalAnchors + outAnchor] = (ry * 2f + gy) * stride;
                            if (_kptDims == 3)
                            {
                                int vCh = k * _kptDims + 2;
                                float rv = lmB[(long)vCh * numAnchorsScale + anchorIdx];
                                outPtr[outBase + (long)vCh * totalAnchors + outAnchor] = 1f / (1f + MathF.Exp(-rv));
                            }
                        }
                    }
                }
            }
            anchorOffset += numAnchorsScale;
        }

        for (int s = 0; s < _numScales; s++) lmLogits[s].Dispose();
        return (detections, landmarks);
    }
}
