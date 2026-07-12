using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.Blocks;

/// <summary>YOLO11-pose head — the full <see cref="DetectHeadV11"/> (box cv2 + class cv3, decoded to
/// <c>[B, 4+nc, A]</c>) plus a keypoint branch (<c>cv4</c>) decoded to <c>[B, nk·ndim, A]</c>. Ultralytics
/// <c>Pose</c> adds, per detect scale, <c>cv4 = Conv(c_in→c4, 3×3) → Conv(c4→c4, 3×3) → Conv2d(c4→nk·ndim, 1×1)</c>
/// with <c>c4 = max(c_in[0]/4, nk·ndim)</c>; the raw keypoint outputs are decoded per anchor as
/// <c>x = (2·rx + gx)·stride</c>, <c>y = (2·ry + gy)·stride</c>, <c>v = σ(rv)</c> — matching
/// <c>Pose.kpts_decode</c> (anchor grid-cell offset 0.5, so <c>anchors − 0.5 = grid index</c>).
///
/// <para>The box/class path is delegated to a composed <see cref="DetectHeadV11"/> so the DFL/anchor/sigmoid
/// decode is not duplicated; this class owns only the keypoint branch + its decode.</para></summary>
public sealed unsafe class PoseHeadV11
{
    private readonly DetectHeadV11 _detect;
    private readonly int _numKpt;      // keypoints per detection (17)
    private readonly int _kptDims;     // values per keypoint (3 = x, y, v)
    private readonly int _nkFlat;      // _numKpt * _kptDims (51)
    private readonly int _c4;          // keypoint-branch hidden width
    private readonly int[] _inChannels;
    private readonly int[] _strides;
    private readonly int _numScales;

    private readonly ConvBnSilu[][] _cv4;    // _cv4[scale][0..1]
    private readonly ConvBnSilu[] _cv4Final; // _cv4Final[scale]

    public int NumClasses => _detect.NumClasses;
    public int NumKeypoints => _numKpt;
    public int KptDims => _kptDims;
    public IReadOnlyList<int> Strides => _strides;

    public PoseHeadV11(int numClasses, int regMax, int numKeypoints, int kptDims, int[] inChannels, int[] strides)
    {
        ArgumentNullException.ThrowIfNull(inChannels);
        ArgumentNullException.ThrowIfNull(strides);
        if (numKeypoints <= 0 || kptDims <= 0)
            throw new ArgumentOutOfRangeException(nameof(numKeypoints), "numKeypoints and kptDims must be positive.");

        _detect = new DetectHeadV11(numClasses, regMax, inChannels, strides);
        _numKpt = numKeypoints;
        _kptDims = kptDims;
        _nkFlat = numKeypoints * kptDims;
        _inChannels = (int[])inChannels.Clone();
        _strides = (int[])strides.Clone();
        _numScales = inChannels.Length;
        _c4 = Math.Max(_inChannels[0] / 4, _nkFlat);

        _cv4 = new ConvBnSilu[_numScales][];
        _cv4Final = new ConvBnSilu[_numScales];
        for (int s = 0; s < _numScales; s++)
        {
            _cv4[s] = new ConvBnSilu[2];
            _cv4[s][0] = new ConvBnSilu(_c4, 1, 1, 1, 1, useSilu: true);   // 3×3, pad 1 (kernel inferred from weights)
            _cv4[s][1] = new ConvBnSilu(_c4, 1, 1, 1, 1, useSilu: true);   // 3×3, pad 1
            _cv4Final[s] = new ConvBnSilu(_nkFlat, 1, 1, 0, 0, useSilu: false);   // 1×1 plain Conv2d (no BN)
        }
    }

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

    /// <summary>Returns the box/class detections <c>[B, 4+nc, A]</c> (input-pixel xywh + sigmoid probs) and the
    /// decoded keypoints <c>[B, nk·ndim, A]</c> (x, y in input pixels, visibility sigmoid'd), both over the same
    /// concatenated anchor order (P3, P4, P5 row-major). Caller owns disposal of both.</summary>
    public (Tensor detections, Tensor keypoints) Forward(IBackend backend, IReadOnlyList<Tensor> features)
    {
        ArgumentNullException.ThrowIfNull(features);
        if (features.Count != _numScales)
            throw new ArgumentException($"Pose head expects {_numScales} feature tensors; got {features.Count}.");

        Tensor detections = _detect.Forward(backend, features);
        int batch = (int)features[0].Shape[0];

        Tensor[] kptLogits = new Tensor[_numScales];
        int[] heights = new int[_numScales];
        int[] widths = new int[_numScales];
        int totalAnchors = 0;
        for (int s = 0; s < _numScales; s++)
        {
            Tensor feat = features[s];
            heights[s] = (int)feat.Shape[2];
            widths[s] = (int)feat.Shape[3];
            totalAnchors += heights[s] * widths[s];
            Tensor k1 = _cv4[s][0].Forward(backend, feat);
            Tensor k2 = _cv4[s][1].Forward(backend, k1);
            k1.Dispose();
            kptLogits[s] = _cv4Final[s].Forward(backend, k2);
            k2.Dispose();
        }

        Tensor keypoints = new Tensor(new TensorShape(batch, _nkFlat, totalAnchors), DType.F32);
        float* outPtr = (float*)keypoints.DataPointer;
        long outBatchStride = (long)_nkFlat * totalAnchors;

        int anchorOffset = 0;
        for (int s = 0; s < _numScales; s++)
        {
            int h = heights[s], w = widths[s];
            int numAnchorsScale = h * w;
            float stride = _strides[s];
            float* kptPtr = (float*)kptLogits[s].DataPointer;
            int kptChStride = _nkFlat * numAnchorsScale;

            for (int bIdx = 0; bIdx < batch; bIdx++)
            {
                float* kptB = kptPtr + bIdx * kptChStride;
                for (int gy = 0; gy < h; gy++)
                {
                    for (int gx = 0; gx < w; gx++)
                    {
                        int anchorIdx = gy * w + gx;
                        long outAnchor = anchorOffset + anchorIdx;
                        long outBase = bIdx * outBatchStride;
                        for (int k = 0; k < _numKpt; k++)
                        {
                            int xCh = k * _kptDims + 0;
                            int yCh = k * _kptDims + 1;
                            float rx = kptB[(long)xCh * numAnchorsScale + anchorIdx];
                            float ry = kptB[(long)yCh * numAnchorsScale + anchorIdx];
                            outPtr[outBase + (long)xCh * totalAnchors + outAnchor] = (rx * 2f + gx) * stride;
                            outPtr[outBase + (long)yCh * totalAnchors + outAnchor] = (ry * 2f + gy) * stride;
                            if (_kptDims == 3)
                            {
                                int vCh = k * _kptDims + 2;
                                float rv = kptB[(long)vCh * numAnchorsScale + anchorIdx];
                                outPtr[outBase + (long)vCh * totalAnchors + outAnchor] = 1f / (1f + MathF.Exp(-rv));
                            }
                        }
                    }
                }
            }
            anchorOffset += numAnchorsScale;
        }

        for (int s = 0; s < _numScales; s++) kptLogits[s].Dispose();
        return (detections, keypoints);
    }
}
