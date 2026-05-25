using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Vision.Detection.Blocks;

/// <summary>YOLOv8 decoupled detection head with DFL (Distribution Focal Loss) decoding.
/// <para>Per scale (P3/P4/P5):</para>
/// <list type="bullet">
///   <item><b>cv2 (box)</b>: <c>ConvBnSilu(in→c2, 3×3) → ConvBnSilu(c2→c2, 3×3) → Conv2D(c2→4*reg_max, 1×1)</c> — last conv has no BN/SiLU.</item>
///   <item><b>cv3 (cls)</b>: <c>ConvBnSilu(in→c3, 3×3) → ConvBnSilu(c3→c3, 3×3) → Conv2D(c3→nc, 1×1)</c> — last conv has no BN/SiLU.</item>
/// </list>
/// <para>Decoding (inference mode):</para>
/// <list type="number">
///   <item>DFL: softmax(reg_max bins) → weighted sum with [0..reg_max-1] → 4 distance scalars (l, t, r, b).</item>
///   <item>Anchor points: grid centers <c>(x + 0.5, y + 0.5)</c> per scale, multiplied by stride at the end.</item>
///   <item>Distances → xywh box: <c>x1y1 = anchor - lt</c>, <c>x2y2 = anchor + rb</c>, then <c>(cx, cy, w, h) = ((x1y1+x2y2)/2, x2y2-x1y1)</c>.</item>
///   <item>Multiply by stride to land in input-pixel coordinates.</item>
///   <item>Sigmoid on class logits.</item>
///   <item>Concat <c>[xywh (4ch), cls (nc ch)]</c> and flatten spatial → <c>[B, 4 + nc, total_anchors]</c>.</item>
/// </list></summary>
public sealed unsafe class DetectHead
{
    private readonly int _numClasses;
    private readonly int _regMax;
    private readonly int[] _inChannels;
    private readonly int[] _strides;
    private readonly int _c2; // box-branch hidden width
    private readonly int _c3; // cls-branch hidden width
    private readonly int _numScales;

    // Per-scale branches.
    //   _cv2[scale][0] / [1] = the two 3×3 ConvBnSilu stages
    //   _cv2Final[scale]      = the 1×1 projection conv (no BN/SiLU), output = 4*reg_max
    private readonly ConvBnSilu[][] _cv2;
    private readonly ConvBnSilu[] _cv2Final;
    private readonly ConvBnSilu[][] _cv3;
    private readonly ConvBnSilu[] _cv3Final;

    /// <summary>Number of classes (80 for COCO).</summary>
    public int NumClasses => _numClasses;

    /// <summary>DFL distribution bin count (16 in YOLOv8).</summary>
    public int RegMax => _regMax;

    /// <summary>Per-scale strides — typically <c>[8, 16, 32]</c> for P3/P4/P5.</summary>
    public IReadOnlyList<int> Strides => _strides;

    /// <summary>Creates the detect head for the given per-scale input channel counts and strides.
    /// <paramref name="inChannels"/> length must equal <paramref name="strides"/> length.</summary>
    public DetectHead(int numClasses, int regMax, int[] inChannels, int[] strides)
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

        // Per Ultralytics: c2 = max(16, ch[0]//4, reg_max*4); c3 = max(ch[0], min(nc, 100)).
        _c2 = Math.Max(16, Math.Max(_inChannels[0] / 4, _regMax * 4));
        _c3 = Math.Max(_inChannels[0], Math.Min(_numClasses, 100));

        _cv2 = new ConvBnSilu[_numScales][];
        _cv2Final = new ConvBnSilu[_numScales];
        _cv3 = new ConvBnSilu[_numScales][];
        _cv3Final = new ConvBnSilu[_numScales];

        for (int s = 0; s < _numScales; s++)
        {
            _cv2[s] = new ConvBnSilu[2];
            _cv2[s][0] = new ConvBnSilu(_c2, 1, 1, 1, 1, useSilu: true);
            _cv2[s][1] = new ConvBnSilu(_c2, 1, 1, 1, 1, useSilu: true);
            // Final 1×1 box projection. No BN, no SiLU — direct logits feed into DFL.
            _cv2Final[s] = new ConvBnSilu(4 * _regMax, 1, 1, 0, 0, useSilu: false);

            _cv3[s] = new ConvBnSilu[2];
            _cv3[s][0] = new ConvBnSilu(_c3, 1, 1, 1, 1, useSilu: true);
            _cv3[s][1] = new ConvBnSilu(_c3, 1, 1, 1, 1, useSilu: true);
            _cv3Final[s] = new ConvBnSilu(_numClasses, 1, 1, 0, 0, useSilu: false);
        }
    }

    /// <summary>Loads detect head weights from a checkpoint. Ultralytics keys follow
    /// <c>{prefix}.cv2.{scale}.{stage}.conv.weight</c> and <c>{prefix}.cv3.{scale}.{stage}.conv.weight</c>
    /// where <c>stage</c> ∈ {0, 1, 2}. The DFL conv weights are fixed (0..reg_max-1) — we don't
    /// load them from the checkpoint, we synthesize them.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        for (int s = 0; s < _numScales; s++)
        {
            _cv2[s][0].LoadWeights(weights, $"{prefix}.cv2.{s}.0.conv");
            _cv2[s][1].LoadWeights(weights, $"{prefix}.cv2.{s}.1.conv");
            // The final box projection in Ultralytics is `cv2[s][2]` (a plain Conv2d, not a Conv-BN).
            // After our PT→safetensors conversion both formats look the same: weight + bias.
            _cv2Final[s].LoadWeights(weights, $"{prefix}.cv2.{s}.2");

            _cv3[s][0].LoadWeights(weights, $"{prefix}.cv3.{s}.0.conv");
            _cv3[s][1].LoadWeights(weights, $"{prefix}.cv3.{s}.1.conv");
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
            foreach (Tensor t in _cv3[s][0].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv3[s][1].EnumerateWeights()) yield return t;
            foreach (Tensor t in _cv3Final[s].EnumerateWeights()) yield return t;
        }
    }

    /// <summary>Runs the detect head end-to-end. Takes one feature tensor per scale, returns a
    /// single output of shape <c>[B, 4 + numClasses, total_anchors]</c>. The first 4 channels are
    /// box (cx, cy, w, h) in input-pixel coords; the remaining channels are class probabilities
    /// after sigmoid. Caller owns the returned tensor.</summary>
    public Tensor Forward(IBackend backend, IReadOnlyList<Tensor> features)
    {
        ArgumentNullException.ThrowIfNull(features);
        if (features.Count != _numScales)
            throw new ArgumentException($"Detect head expects {_numScales} feature tensors; got {features.Count}.");

        int batch = (int)features[0].Shape[0];

        // Step 1: per-scale logits. Each branch outputs raw box (4*reg_max ch) and raw class (nc ch),
        // both [B, ch, H_s, W_s].
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

            Tensor b1 = _cv2[s][0].Forward(backend, feat);
            Tensor b2 = _cv2[s][1].Forward(backend, b1);
            b1.Dispose();
            boxLogits[s] = _cv2Final[s].Forward(backend, b2);
            b2.Dispose();

            Tensor c1 = _cv3[s][0].Forward(backend, feat);
            Tensor c2 = _cv3[s][1].Forward(backend, c1);
            c1.Dispose();
            clsLogits[s] = _cv3Final[s].Forward(backend, c2);
            c2.Dispose();
        }

        // Step 2: decode per scale and merge into one [B, 4 + nc, total_anchors] tensor.
        Tensor output = new Tensor(new TensorShape(batch, 4 + _numClasses, totalAnchors), DType.F32);
        float* outPtr = (float*)output.DataPointer;

        // Scratch buffers allocated once across the entire decode loop. Stackalloc inside the
        // inner loop would warn CA2014 (stack overflow risk) and re-allocate per anchor; this
        // form reuses 4 + reg_max float slots for the lifetime of the call.
        Span<float> distances = stackalloc float[4];
        Span<float> probs = stackalloc float[_regMax];

        int anchorOffset = 0;
        for (int s = 0; s < _numScales; s++)
        {
            int h = heights[s];
            int w = widths[s];
            int numAnchorsScale = h * w;
            float stride = _strides[s];

            float* boxPtr = (float*)boxLogits[s].DataPointer; // [B, 4*reg_max, h*w]
            float* clsPtr = (float*)clsLogits[s].DataPointer; // [B, nc, h*w]
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
                        // DFL: for each of 4 distance values, softmax over reg_max bins then take
                        // weighted sum with bin index. Indices in box tensor: channel = c*reg_max + bin.
                        // Source layout is [4*reg_max channels × (h*w)] flat per batch.
                        for (int dim = 0; dim < 4; dim++)
                        {
                            // Find max for numerical stability of softmax.
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
                        float x1 = anchorX - l;
                        float y1 = anchorY - t;
                        float x2 = anchorX + r;
                        float y2 = anchorY + btm;
                        float cx = (x1 + x2) * 0.5f * stride;
                        float cy = (y1 + y2) * 0.5f * stride;
                        float bw = (x2 - x1) * stride;
                        float bh = (y2 - y1) * stride;

                        // Write into output [B, 4+nc, total_anchors]. Channel-major layout.
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
