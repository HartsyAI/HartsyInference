using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Wan-Animate motion encoder (<c>WanAnimateMotionEncoder</c> in diffusers <c>transformer_wan_animate.py</c>):
/// a StyleGAN-style appearance encoder that maps a face frame <c>[B, 3, size, size]</c> to a motion vector
/// <c>[B, out_dim]</c> via "Linear Motion Decomposition" — conv stem + residual down-blocks → <c>conv_out</c> (1×1
/// spatial) → motion MLP → <c>motion_feat @ Qᵀ</c> where <c>Q</c> is the orthonormal basis from a QR of the learned
/// <c>motion_synthesis_weight [out_dim, motion_dim]</c>.
///
/// <para>Building blocks (nested): <c>FusedLeakyReLU</c> (<c>leaky_relu(x+bias)·√2</c>), <c>MotionConv2d</c>
/// (optional depthwise FIR blur → scaled Conv2d → fused act), <c>MotionLinear</c> (scaled linear), and the residual
/// down-block (<c>(conv2(conv1(x)) + conv_skip(x))/√2</c>). The QR is done via modified Gram-Schmidt in FP32
/// (structural stand-in for <c>torch.linalg.qr</c>; column-sign conventions may differ — validation-gated).</para></summary>
public sealed unsafe class WanAnimateMotionEncoder
{
    // Reference channel table keyed by spatial resolution (WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES).
    private static readonly Dictionary<int, int> _channelSizes = new()
    {
        [4] = 512, [8] = 512, [16] = 512, [32] = 512, [64] = 256, [128] = 128, [256] = 64, [512] = 32, [1024] = 16,
    };

    private readonly int _size, _styleDim, _motionDim, _outDim, _motionBlocks;
    private readonly Dictionary<int, int> _channels;

    private MotionConv2d? _convIn, _convOut;
    private MotionResBlock[] _resBlocks = [];
    private MotionLinear[] _motionNetwork = [];
    private Tensor? _motionSynthesisWeight;   // [out_dim, motion_dim]

    public WanAnimateMotionEncoder(int size = 512, int styleDim = 512, int motionDim = 20, int outDim = 512,
        int motionBlocks = 5, Dictionary<int, int>? channels = null)
    {
        _size = size;
        _styleDim = styleDim;
        _motionDim = motionDim;
        _outDim = outDim;
        _motionBlocks = motionBlocks;
        _channels = channels ?? _channelSizes;
        if (!_channels.ContainsKey(size))
            throw new ArgumentException($"motion-encoder channel table has no entry for size {size}.", nameof(size));
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _convIn = new MotionConv2d(); _convIn.LoadWeights(w, $"{p}.conv_in", useActivation: true); _convIn.Configure(stride: 1, pad: 0, blur: false);

        int logSize = (int)Math.Round(Math.Log2(_size));
        List<MotionResBlock> blocks = new();
        int idx = 0;
        for (int i = logSize; i > 2; i--)
        {
            MotionResBlock b = new();
            b.LoadWeights(w, $"{p}.res_blocks.{idx}");
            blocks.Add(b);
            idx++;
        }
        _resBlocks = blocks.ToArray();

        _convOut = new MotionConv2d(); _convOut.LoadWeights(w, $"{p}.conv_out", useActivation: false); _convOut.Configure(stride: 1, pad: 0, blur: false);

        _motionNetwork = new MotionLinear[_motionBlocks];
        for (int i = 0; i < _motionBlocks; i++)
        {
            _motionNetwork[i] = new MotionLinear();
            _motionNetwork[i].LoadWeights(w, $"{p}.motion_network.{i}");
        }
        _motionSynthesisWeight = LoadF32(w, $"{p}.motion_synthesis_weight");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convIn is not null) foreach (Tensor t in _convIn.EnumerateWeights()) yield return t;
        foreach (MotionResBlock b in _resBlocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        if (_convOut is not null) foreach (Tensor t in _convOut.EnumerateWeights()) yield return t;
        foreach (MotionLinear l in _motionNetwork) foreach (Tensor t in l.EnumerateWeights()) yield return t;
        if (_motionSynthesisWeight is not null) yield return _motionSynthesisWeight;
    }

    /// <summary>Encodes a batch of face frames <c>[B, 3, size, size]</c> → motion vectors <c>[B, out_dim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor faceImage)
    {
        int b = (int)faceImage.Shape[0];
        if ((int)faceImage.Shape[2] != _size || (int)faceImage.Shape[3] != _size)
            throw new ArgumentException($"face image must be {_size}×{_size}; got {faceImage.Shape[2]}×{faceImage.Shape[3]}.");

        Tensor h = _convIn!.Forward(backend, faceImage);
        foreach (MotionResBlock blk in _resBlocks) { Tensor n = blk.Forward(backend, h); h.Dispose(); h = n; }
        Tensor convOut = _convOut!.Forward(backend, h);   // [B, style_dim, 1, 1]
        h.Dispose();

        // squeeze spatial → [B, style_dim]
        Tensor motionFeat = new Tensor(new TensorShape(b, _styleDim), DType.F32);
        Buffer.MemoryCopy((float*)convOut.DataPointer, (float*)motionFeat.DataPointer, (long)b * _styleDim * 4, (long)b * _styleDim * 4);
        convOut.Dispose();

        foreach (MotionLinear l in _motionNetwork) { Tensor n = l.Forward(backend, motionFeat); motionFeat.Dispose(); motionFeat = n; }
        // motionFeat is now [B, motion_dim]

        // Linear Motion Decomposition: Q = QR(weight)[0] (economy, orthonormal columns); motion_vec = motion_feat @ Qᵀ.
        Tensor qt = OrthonormalBasisTransposed(_motionSynthesisWeight!, _outDim, _motionDim);   // [motion_dim, out_dim]
        Tensor motionVec = new Tensor(new TensorShape(b, _outDim), DType.F32);
        backend.MatMul(motionVec, motionFeat, qt);
        motionFeat.Dispose();
        qt.Dispose();
        return motionVec;
    }

    /// <summary>Modified Gram-Schmidt on the columns of <paramref name="weight"/> <c>[out_dim, motion_dim]</c> (+1e-8),
    /// returning <c>Qᵀ</c> <c>[motion_dim, out_dim]</c> (orthonormal columns of Q as rows) for the <c>x @ Qᵀ</c> product.</summary>
    private static Tensor OrthonormalBasisTransposed(Tensor weight, int m, int n)
    {
        Tensor wf = weight.DType == DType.F32 ? weight : weight.CastTo(DType.F32);
        float* wp = (float*)wf.DataPointer;
        // Column-major copy of W into q[col][row].
        float[][] q = new float[n][];
        for (int j = 0; j < n; j++)
        {
            q[j] = new float[m];
            for (int r = 0; r < m; r++) q[j][r] = wp[(long)r * n + j] + 1e-8f;
        }
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < j; i++)
            {
                double dot = 0; for (int r = 0; r < m; r++) dot += (double)q[i][r] * q[j][r];
                for (int r = 0; r < m; r++) q[j][r] -= (float)(dot * q[i][r]);
            }
            double norm = 0; for (int r = 0; r < m; r++) norm += (double)q[j][r] * q[j][r];
            float inv = (float)(1.0 / Math.Sqrt(norm + 1e-12));
            for (int r = 0; r < m; r++) q[j][r] *= inv;
        }
        // Qᵀ: [n, m] with row j = column j of Q.
        Tensor qt = new Tensor(new TensorShape(n, m), DType.F32);
        float* qp = (float*)qt.DataPointer;
        for (int j = 0; j < n; j++)
            for (int r = 0; r < m; r++) qp[(long)j * m + r] = q[j][r];
        if (!ReferenceEquals(wf, weight)) wf.Dispose();
        return qt;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key) { Tensor t = w[key]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }

    /// <summary>Fused leaky-ReLU with channel-wise bias: <c>leaky_relu(x + bias, 0.2) · √2</c>, bias over channel dim 1.</summary>
    private static void FusedLeakyReLU(Tensor x, Tensor? bias, int channels)
    {
        const float slope = 0.2f, scale = 1.41421356f;
        int b = (int)x.Shape[0];
        long spatial = x.Shape.ElementCount / ((long)b * channels);
        float* xp = (float*)x.DataPointer;
        float* bp = bias is null ? null : (float*)bias.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (int c = 0; c < channels; c++)
            {
                float biasVal = bp is null ? 0f : bp[c];
                long basePos = ((long)bi * channels + c) * spatial;
                for (long s = 0; s < spatial; s++)
                {
                    float v = xp[basePos + s] + biasVal;
                    xp[basePos + s] = (v >= 0f ? v : v * slope) * scale;
                }
            }
    }

    /// <summary>StyleGAN-style scaled Conv2d with optional depthwise FIR blur (kernel <c>(1,3,3,1)</c> outer product)
    /// and a fused leaky-ReLU. Weight is runtime-scaled by <c>1/√(in·k²)</c>.</summary>
    private sealed class MotionConv2d
    {
        private Tensor? _weight, _bias;     // [outC, inC, k, k]
        private bool _useAct, _blur;
        private int _inC, _outC, _k, _stride, _pad, _blurPad;
        private float _scale;
        private float[]? _blurKernel;       // normalized [4,4] flattened

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p, bool useActivation)
        {
            _weight = LoadF32(w, $"{p}.weight");
            _outC = (int)_weight.Shape[0]; _inC = (int)_weight.Shape[1]; _k = (int)_weight.Shape[2];
            _scale = 1f / MathF.Sqrt(_inC * _k * _k);
            _useAct = useActivation;
            if (useActivation) w.TryGetValue($"{p}.act_fn.bias", out _bias);
            else w.TryGetValue($"{p}.bias", out _bias);
            // Stride/pad/blur are derivable from the conv role: encode them from the saved shape conventions.
            // (Set externally per the reference wiring — defaults below cover conv_in/conv_out; res-block convs set them.)
        }

        public void Configure(int stride, int pad, bool blur)
        {
            _stride = stride; _pad = pad; _blur = blur;
            if (blur)
            {
                float[] k1 = [1, 3, 3, 1];
                float sum = 0; foreach (float v in k1) sum += v; sum *= sum;   // 2D sum = (Σk)² = 64
                _blurKernel = new float[16];
                for (int a = 0; a < 4; a++) for (int bb = 0; bb < 4; bb++) _blurKernel[a * 4 + bb] = k1[a] * k1[bb] / sum;
                int pPad = (4 - stride) + (_k - 1);
                _blurPad = (pPad + 1) / 2;   // symmetric stand-in for ((p+1)//2, p//2)
            }
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            if (_weight is not null) yield return _weight;
            if (_bias is not null) yield return _bias;
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            Tensor cur = x;
            bool ownCur = false;
            if (_blur && _blurKernel is not null)
            {
                Tensor blurred = DepthwiseBlur(cur);
                cur = blurred; ownCur = true;
            }
            // Scaled main conv (scale folded into a temp weight copy).
            Tensor scaledW = new Tensor(_weight!.Shape, DType.F32);
            long wn = _weight.Shape.ElementCount;
            float* swp = (float*)scaledW.DataPointer, wp = (float*)_weight.DataPointer;
            for (long i = 0; i < wn; i++) swp[i] = wp[i] * _scale;
            int h = (int)cur.Shape[2], wdt = (int)cur.Shape[3];
            int oh = (h + 2 * _pad - _k) / _stride + 1, ow = (wdt + 2 * _pad - _k) / _stride + 1;
            Tensor o = new Tensor(new TensorShape((int)cur.Shape[0], _outC, oh, ow), DType.F32);
            backend.Conv2D(o, cur, scaledW, _useAct ? null : _bias, _stride, _stride, _pad, _pad);
            scaledW.Dispose();
            if (ownCur) cur.Dispose();
            if (_useAct) FusedLeakyReLU(o, _bias, _outC);
            return o;
        }

        /// <summary>Depthwise FIR blur (4×4 normalized kernel, symmetric pad, stride 1) — a manual per-channel conv
        /// since <see cref="IBackend.Conv2D"/> has no grouped mode.</summary>
        private Tensor DepthwiseBlur(Tensor x)
        {
            int b = (int)x.Shape[0], c = (int)x.Shape[1], h = (int)x.Shape[2], wdt = (int)x.Shape[3];
            int pad = _blurPad;
            int oh = h + 2 * pad - 4 + 1, ow = wdt + 2 * pad - 4 + 1;
            Tensor o = new Tensor(new TensorShape(b, c, oh, ow), DType.F32);
            float* xp = (float*)x.DataPointer; float* op = (float*)o.DataPointer; float* kp;
            fixed (float* kfix = _blurKernel)
            {
                kp = kfix;
                for (int bi = 0; bi < b; bi++)
                    for (int ci = 0; ci < c; ci++)
                    {
                        long inBase = ((long)bi * c + ci) * h * wdt;
                        long outBase = ((long)bi * c + ci) * oh * ow;
                        for (int y = 0; y < oh; y++)
                            for (int xo = 0; xo < ow; xo++)
                            {
                                float acc = 0;
                                for (int ky = 0; ky < 4; ky++)
                                {
                                    int iy = y + ky - pad;
                                    if (iy < 0 || iy >= h) continue;
                                    for (int kx = 0; kx < 4; kx++)
                                    {
                                        int ix = xo + kx - pad;
                                        if (ix < 0 || ix >= wdt) continue;
                                        acc += xp[inBase + (long)iy * wdt + ix] * kp[ky * 4 + kx];
                                    }
                                }
                                op[outBase + (long)y * ow + xo] = acc;
                            }
                    }
            }
            return o;
        }
    }

    /// <summary>StyleGAN-style scaled linear with optional fused leaky-ReLU (<c>1/√in</c> weight scale).</summary>
    private sealed class MotionLinear
    {
        private Tensor? _weight, _bias;
        private bool _useAct;
        private int _inDim, _outDim;
        private float _scale;

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p, bool useActivation = false)
        {
            _weight = LoadF32(w, $"{p}.weight");
            _outDim = (int)_weight.Shape[0]; _inDim = (int)_weight.Shape[1];
            _scale = 1f / MathF.Sqrt(_inDim);
            _useAct = useActivation;
            if (useActivation) w.TryGetValue($"{p}.act_fn.bias", out _bias);
            else w.TryGetValue($"{p}.bias", out _bias);
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            if (_weight is not null) yield return _weight;
            if (_bias is not null) yield return _bias;
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            int rows = (int)x.Shape[0];
            Tensor scaledW = new Tensor(_weight!.Shape, DType.F32);
            long wn = _weight.Shape.ElementCount;
            float* swp = (float*)scaledW.DataPointer, wp = (float*)_weight.DataPointer;
            for (long i = 0; i < wn; i++) swp[i] = wp[i] * _scale;
            Tensor o = new Tensor(new TensorShape(rows, _outDim), DType.F32);
            backend.Linear(o, x, scaledW, _useAct ? null : _bias);
            scaledW.Dispose();
            if (_useAct) FusedLeakyReLU(o, _bias, _outDim);
            return o;
        }
    }

    /// <summary>Motion-encoder residual down-block: <c>(conv2(conv1(x)) + conv_skip(x)) / √2</c>.</summary>
    private sealed class MotionResBlock
    {
        private readonly MotionConv2d _conv1 = new(), _conv2 = new(), _convSkip = new();

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _conv1.LoadWeights(w, $"{p}.conv1", useActivation: true); _conv1.Configure(stride: 1, pad: 1, blur: false);
            _conv2.LoadWeights(w, $"{p}.conv2", useActivation: true); _conv2.Configure(stride: 2, pad: 0, blur: true);
            _convSkip.LoadWeights(w, $"{p}.conv_skip", useActivation: false); _convSkip.Configure(stride: 2, pad: 0, blur: true);
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor t in _conv1.EnumerateWeights()) yield return t;
            foreach (Tensor t in _conv2.EnumerateWeights()) yield return t;
            foreach (Tensor t in _convSkip.EnumerateWeights()) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            Tensor a = _conv1.Forward(backend, x);
            Tensor b = _conv2.Forward(backend, a); a.Dispose();
            Tensor skip = _convSkip.Forward(backend, x);
            long n = b.Shape.ElementCount;
            float* bp = (float*)b.DataPointer, sp = (float*)skip.DataPointer;
            float inv = 1f / 1.41421356f;
            for (long i = 0; i < n; i++) bp[i] = (bp[i] + sp[i]) * inv;
            skip.Dispose();
            return b;
        }
    }
}
