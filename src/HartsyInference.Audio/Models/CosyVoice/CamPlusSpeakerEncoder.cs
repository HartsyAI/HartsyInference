using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>CAM++ speaker encoder (3D-Speaker <c>campplus_cn_common</c>) producing the speaker embedding
/// CosyVoice2 feeds to its flow-matching stage. Faithful to the upstream <c>CAMPPlus</c>: an FCM ResNet front
/// over the [B,1,freq,time] fbank (all strides reduce frequency 80→10, time preserved) → a TDNN stem →
/// three DenseTDNN blocks (12 / 24 / 16 <c>CAMDenseTDNNLayer</c>s, each BN+ReLU → 1×1 bottleneck → BN+ReLU →
/// context-aware <c>CAMLayer</c>) with transition layers between → BN+ReLU → statistics pooling → a dense
/// projection to the embedding (no L2-norm; matches the model's raw output).
///
/// <para>Input is an 80-bin fbank <c>[1, T, 80]</c>; output is <c>[1, embedding]</c> (192 for the common model).
/// Weights load from the upstream PyTorch checkpoint's clean names (<c>head.*</c> / <c>xvector.*</c>) — the same
/// weights CosyVoice2 ships as <c>campplus.onnx</c>, so they validate against an ONNX-Runtime reference. Convs
/// route through <see cref="IBackend"/>; BatchNorm / pooling / CAM gating are CPU sweeps (the encoder is a
/// one-shot reference pass, not a hot loop).</para></summary>
public sealed unsafe class CamPlusSpeakerEncoder : IDisposable
{
    private const float BnEps = 1e-5f;
    private const int FeatDim = 80;
    private const int FcmCh = 32;
    private const int Growth = 32;
    private const int BnCh = 128;       // bn_size(4) * growth(32)
    private static readonly int[] BlockLayers = [12, 24, 16];
    private static readonly int[] BlockDilations = [1, 2, 2];

    private readonly int _embedDim;
    private int _disposed;

    // FCM (ResNet front). layer1/layer2 each have 2 BasicResBlocks; the first of each downsamples freq.
    private Conv2dW _fcmConv1 = new(); private Bn _fcmBn1 = new();
    private ResBlock[] _fcmLayer1 = []; private ResBlock[] _fcmLayer2 = [];
    private Conv2dW _fcmConv2 = new(); private Bn _fcmBn2 = new();

    // TDNN stem.
    private Tensor? _tdnnW; private Bn _tdnnBn = new();
    // Dense blocks + transitions.
    private DenseLayerW[][] _blocks = [];
    private (Bn Norm, Tensor? W)[] _transit = [];
    private Bn _outBn = new();
    // Final dense (BN affine=false, no ReLU).
    private Tensor? _denseW; private Bn _denseBn = new();

    public CamPlusSpeakerEncoder(int embedDim = 192) => _embedDim = embedDim;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";

        _fcmConv1 = Conv2dW.Load(w, $"{p}head.conv1");
        _fcmBn1 = Bn.Load(w, $"{p}head.bn1", affine: true);
        _fcmLayer1 = [ResBlock.Load(w, $"{p}head.layer1.0"), ResBlock.Load(w, $"{p}head.layer1.1")];
        _fcmLayer2 = [ResBlock.Load(w, $"{p}head.layer2.0"), ResBlock.Load(w, $"{p}head.layer2.1")];
        _fcmConv2 = Conv2dW.Load(w, $"{p}head.conv2");
        _fcmBn2 = Bn.Load(w, $"{p}head.bn2", affine: true);

        _tdnnW = WhisperOps.EnsureF32(w[$"{p}xvector.tdnn.linear.weight"]);
        _tdnnBn = Bn.Load(w, $"{p}xvector.tdnn.nonlinear.batchnorm", affine: true);

        _blocks = new DenseLayerW[BlockLayers.Length][];
        _transit = new (Bn, Tensor?)[BlockLayers.Length];
        for (int b = 0; b < BlockLayers.Length; b++)
        {
            _blocks[b] = new DenseLayerW[BlockLayers[b]];
            for (int i = 0; i < BlockLayers[b]; i++)
                _blocks[b][i] = DenseLayerW.Load(w, $"{p}xvector.block{b + 1}.tdnnd{i + 1}");
            _transit[b] = (Bn.Load(w, $"{p}xvector.transit{b + 1}.nonlinear.batchnorm", affine: true),
                WhisperOps.EnsureF32(w[$"{p}xvector.transit{b + 1}.linear.weight"]));
        }
        _outBn = Bn.Load(w, $"{p}xvector.out_nonlinear.batchnorm", affine: true);
        _denseW = WhisperOps.EnsureF32(w[$"{p}xvector.dense.linear.weight"]);
        _denseBn = Bn.Load(w, $"{p}xvector.dense.nonlinear.batchnorm", affine: false);
    }

    /// <summary>Extracts the raw speaker embedding <c>[1, embedDim]</c> from an 80-bin fbank <c>[1, T, 80]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor fbank)
    {
        if (_tdnnW is null) throw new InvalidOperationException("CamPlusSpeakerEncoder weights not loaded.");
        int t = (int)fbank.Shape[1];

        // (B,T,F) → (B,F,T) → view as NCHW [1,1,80,T]: FCM convs treat H=freq, W=time.
        Tensor ft = new(new TensorShape(1, FeatDim, t), DType.F32);
        backend.Transpose2D(ft, fbank, t, FeatDim);
        Tensor x = ft.Reshape(new TensorShape(1, 1, FeatDim, t));

        x = Relu(backend, Bn2d(Conv2d(backend, x, _fcmConv1, 1, 1, 1, 1), _fcmBn1));
        foreach (ResBlock rb in _fcmLayer1) x = rb.Forward(backend, x);
        foreach (ResBlock rb in _fcmLayer2) x = rb.Forward(backend, x);
        x = Relu(backend, Bn2d(Conv2d(backend, x, _fcmConv2, 2, 1, 1, 1), _fcmBn2));

        // [1,32,10,T] → [1,320,T] (contiguous reshape: channel-major × freq).
        int fcmC = (int)x.Shape[1], fcmF = (int)x.Shape[2], fcmT = (int)x.Shape[3];
        Tensor seq = x.Reshape(new TensorShape(1, fcmC * fcmF, fcmT));

        // TDNN stem: Conv1d(320,128,k5,s2,p2) + BN + ReLU.
        Tensor h = Conv1d(backend, seq, _tdnnW!, null, stride: 2, pad: 2, dil: 1);
        seq.Dispose();
        Bn1d(h, _tdnnBn); Relu(backend, h);

        for (int b = 0; b < _blocks.Length; b++)
        {
            foreach (DenseLayerW layer in _blocks[b])
            {
                Tensor outL = layer.Forward(backend, h, BlockDilations[b]);
                Tensor cat = Concat(h, outL);     // dense growth: cat([x, layer(x)], dim=1)
                h.Dispose(); outL.Dispose();
                h = cat;
            }
            // Transition: BN+ReLU → Conv1d(C, C/2, 1, bias=false).
            Bn1d(h, _transit[b].Norm); Relu(backend, h);
            Tensor tr = Conv1d(backend, h, _transit[b].W!, null, 1, 0, 1);
            h.Dispose(); h = tr;
        }

        Bn1d(h, _outBn); Relu(backend, h);

        // StatsPool: per-channel mean + unbiased std over time → [1, 2C].
        Tensor stats = StatsPool(h);
        h.Dispose();

        // Dense: Conv1d(2C, embed, 1, bias=false) on [1,2C,1] + BN(affine=false), no ReLU.
        Tensor stats3 = stats.Reshape(new TensorShape(1, (int)stats.Shape[1], 1));
        Tensor emb = Conv1d(backend, stats3, _denseW!, null, 1, 0, 1);
        stats.Dispose();
        Bn1d(emb, _denseBn);
        return emb.Reshape(new TensorShape(1, _embedDim));
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _fcmConv1.Weights()) yield return t;
        foreach (Tensor t in _fcmBn1.Weights()) yield return t;
        foreach (ResBlock r in _fcmLayer1) foreach (Tensor t in r.AllWeights()) yield return t;
        foreach (ResBlock r in _fcmLayer2) foreach (Tensor t in r.AllWeights()) yield return t;
        foreach (Tensor t in _fcmConv2.Weights()) yield return t;
        foreach (Tensor t in _fcmBn2.Weights()) yield return t;
        if (_tdnnW is not null) yield return _tdnnW;
        foreach (Tensor t in _tdnnBn.Weights()) yield return t;
        foreach (DenseLayerW[] blk in _blocks) foreach (DenseLayerW l in blk) foreach (Tensor t in l.AllWeights()) yield return t;
        foreach ((Bn norm, Tensor? tw) in _transit) { foreach (Tensor t in norm.Weights()) yield return t; if (tw is not null) yield return tw; }
        foreach (Tensor t in _outBn.Weights()) yield return t;
        if (_denseW is not null) yield return _denseW;
        foreach (Tensor t in _denseBn.Weights()) yield return t;
    }

    // ---- shared op helpers ----

    private static Tensor Conv2d(IBackend backend, Tensor x, Conv2dW w, int sH, int sW, int pH, int pW)
    {
        int outC = (int)w.W!.Shape[0], kh = (int)w.W.Shape[2], kw = (int)w.W.Shape[3];
        int H = (int)x.Shape[2], W = (int)x.Shape[3];
        int outH = (H + 2 * pH - kh) / sH + 1, outW = (W + 2 * pW - kw) / sW + 1;
        Tensor o = new(new TensorShape(1, outC, outH, outW), DType.F32);
        backend.Conv2D(o, x, w.W, null, sH, sW, pH, pW);
        x.Dispose();
        return o;
    }

    private static Tensor Conv1d(IBackend backend, Tensor x, Tensor w, Tensor? b, int stride, int pad, int dil)
    {
        int outC = (int)w.Shape[0], k = (int)w.Shape[2], T = (int)x.Shape[2];
        int outT = (T + 2 * pad - dil * (k - 1) - 1) / stride + 1;
        Tensor o = new(new TensorShape(1, outC, outT), DType.F32);
        backend.Conv1d(o, x, w, b, stride, pad, pad, dil, 1);
        return o;
    }

    private static Tensor Relu(IBackend backend, Tensor x) { backend.LeakyRelu(x, x, 0f); return x; }

    private static Tensor Bn1d(Tensor x, Bn bn)
    {
        int C = (int)x.Shape[1], T = (int)x.Shape[2];
        float* p = (float*)x.DataPointer;
        float* mean = (float*)bn.Mean!.DataPointer, var = (float*)bn.Var!.DataPointer;
        float* wp = bn.Affine ? (float*)bn.W!.DataPointer : null;
        float* bp = bn.Affine ? (float*)bn.B!.DataPointer : null;
        for (int c = 0; c < C; c++)
        {
            float scale = (wp is null ? 1f : wp[c]) / MathF.Sqrt(var[c] + BnEps);
            float shift = (bp is null ? 0f : bp[c]) - mean[c] * scale;
            long baseI = (long)c * T;
            for (int j = 0; j < T; j++) p[baseI + j] = p[baseI + j] * scale + shift;
        }
        return x;
    }

    private static Tensor Bn2d(Tensor x, Bn bn)
    {
        int C = (int)x.Shape[1]; long hw = (long)x.Shape[2] * (int)x.Shape[3];
        float* p = (float*)x.DataPointer;
        float* mean = (float*)bn.Mean!.DataPointer, var = (float*)bn.Var!.DataPointer;
        float* wp = (float*)bn.W!.DataPointer, bp = (float*)bn.B!.DataPointer;
        for (int c = 0; c < C; c++)
        {
            float scale = wp[c] / MathF.Sqrt(var[c] + BnEps);
            float shift = bp[c] - mean[c] * scale;
            long baseI = c * hw;
            for (long j = 0; j < hw; j++) p[baseI + j] = p[baseI + j] * scale + shift;
        }
        return x;
    }

    /// <summary>Concatenates two <c>[1, C, T]</c> tensors along the channel dim.</summary>
    private static Tensor Concat(Tensor a, Tensor b)
    {
        int ca = (int)a.Shape[1], cb = (int)b.Shape[1], t = (int)a.Shape[2];
        Tensor o = new(new TensorShape(1, ca + cb, t), DType.F32);
        long aBytes = (long)ca * t * 4, bBytes = (long)cb * t * 4;
        Buffer.MemoryCopy((void*)a.DataPointer, (void*)o.DataPointer, aBytes, aBytes);
        Buffer.MemoryCopy((void*)b.DataPointer, (byte*)o.DataPointer + aBytes, bBytes, bBytes);
        return o;
    }

    private Tensor StatsPool(Tensor x)
    {
        int C = (int)x.Shape[1], T = (int)x.Shape[2];
        Tensor o = new(new TensorShape(1, 2 * C), DType.F32);
        float* p = (float*)x.DataPointer, op = (float*)o.DataPointer;
        for (int c = 0; c < C; c++)
        {
            long baseI = (long)c * T;
            double mean = 0;
            for (int j = 0; j < T; j++) mean += p[baseI + j];
            mean /= T;
            double var = 0;
            for (int j = 0; j < T; j++) { double d = p[baseI + j] - mean; var += d * d; }
            var = T > 1 ? var / (T - 1) : 0;     // unbiased std (matches torch std default)
            op[c] = (float)mean;
            op[C + c] = (float)Math.Sqrt(var);
        }
        return o;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    // ---- weight holders ----

    private sealed class Bn
    {
        public Tensor? W, B, Mean, Var;
        public bool Affine;
        public static Bn Load(IReadOnlyDictionary<string, Tensor> w, string p, bool affine) => new()
        {
            Affine = affine,
            W = affine ? WhisperOps.EnsureF32(w[$"{p}.weight"]) : null,
            B = affine ? WhisperOps.EnsureF32(w[$"{p}.bias"]) : null,
            Mean = WhisperOps.EnsureF32(w[$"{p}.running_mean"]),
            Var = WhisperOps.EnsureF32(w[$"{p}.running_var"]),
        };
        public IEnumerable<Tensor> Weights()
        {
            if (W is not null) yield return W; if (B is not null) yield return B;
            if (Mean is not null) yield return Mean!; if (Var is not null) yield return Var!;
        }
    }

    private sealed class Conv2dW
    {
        public Tensor? W;
        public static Conv2dW Load(IReadOnlyDictionary<string, Tensor> w, string p)
            => new() { W = WhisperOps.EnsureF32(w[$"{p}.weight"]) };
        public IEnumerable<Tensor> Weights() { if (W is not null) yield return W; }
    }

    /// <summary>An FCM <c>BasicResBlock</c>: conv1(stride freq)+bn1+relu → conv2+bn2, plus an optional 1×1
    /// shortcut (present only on the downsampling first block of each layer), then residual add + ReLU.</summary>
    private sealed class ResBlock
    {
        private Conv2dW _conv1 = new(), _conv2 = new(), _scConv = new();
        private Bn _bn1 = new(), _bn2 = new(), _scBn = new();
        private bool _hasShortcut;
        private int _strideH;

        public static ResBlock Load(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            ResBlock r = new()
            {
                _conv1 = Conv2dW.Load(w, $"{p}.conv1"),
                _bn1 = Bn.Load(w, $"{p}.bn1", affine: true),
                _conv2 = Conv2dW.Load(w, $"{p}.conv2"),
                _bn2 = Bn.Load(w, $"{p}.bn2", affine: true),
                _hasShortcut = w.ContainsKey($"{p}.shortcut.0.weight"),
            };
            // Downsampling block (shortcut present) uses stride (2,1); otherwise stride 1.
            r._strideH = r._hasShortcut ? 2 : 1;
            if (r._hasShortcut)
            {
                r._scConv = Conv2dW.Load(w, $"{p}.shortcut.0");
                r._scBn = Bn.Load(w, $"{p}.shortcut.1", affine: true);
            }
            return r;
        }

        public Tensor Forward(IBackend backend, Tensor x)
        {
            // conv1 stride (strideH, 1), pad 1; conv2 stride 1, pad 1.
            Tensor identity = x;
            Tensor o = Conv2d(backend, Clone(x), _conv1, _strideH, 1, 1, 1);
            Relu(backend, Bn2d(o, _bn1));
            o = Conv2d(backend, o, _conv2, 1, 1, 1, 1);
            Bn2d(o, _bn2);

            Tensor shortcut;
            if (_hasShortcut)
            {
                shortcut = Conv2d(backend, Clone(identity), _scConv, _strideH, 1, 0, 0);
                Bn2d(shortcut, _scBn);
            }
            else
            {
                shortcut = identity;     // identity (stride 1, in==out)
            }
            AddInPlace(o, shortcut);
            if (_hasShortcut) shortcut.Dispose();
            identity.Dispose();
            Relu(backend, o);
            return o;
        }

        public IEnumerable<Tensor> AllWeights()
        {
            foreach (Tensor t in _conv1.Weights()) yield return t;
            foreach (Tensor t in _bn1.Weights()) yield return t;
            foreach (Tensor t in _conv2.Weights()) yield return t;
            foreach (Tensor t in _bn2.Weights()) yield return t;
            if (_hasShortcut)
            {
                foreach (Tensor t in _scConv.Weights()) yield return t;
                foreach (Tensor t in _scBn.Weights()) yield return t;
            }
        }

        private static Tensor Clone(Tensor x)
        {
            Tensor o = new(x.Shape, DType.F32);
            long bytes = x.ElementCount * 4;
            Buffer.MemoryCopy((void*)x.DataPointer, (void*)o.DataPointer, bytes, bytes);
            return o;
        }

        private static void AddInPlace(Tensor dst, Tensor src)
        {
            float* d = (float*)dst.DataPointer, s = (float*)src.DataPointer;
            for (long i = 0; i < dst.ElementCount; i++) d[i] += s[i];
        }
    }

    /// <summary>A <c>CAMDenseTDNNLayer</c>: BN+ReLU → 1×1 bottleneck (in→128) → BN+ReLU → CAM context-attention
    /// layer (local Conv1d gated by a sigmoid of mean + segment-pooled context) producing <c>growth</c> channels.</summary>
    private sealed class DenseLayerW
    {
        private Bn _norm1 = new(), _norm2 = new();
        private Tensor? _linear1;        // 1×1 bottleneck, in → 128, bias=false
        private Tensor? _localW;         // cam linear_local Conv1d(128, 32, 3), bias=false
        private Tensor? _ctx1W, _ctx1B;  // cam linear1 Conv1d(128, 64, 1)
        private Tensor? _ctx2W, _ctx2B;  // cam linear2 Conv1d(64, 32, 1)

        public static DenseLayerW Load(IReadOnlyDictionary<string, Tensor> w, string p) => new()
        {
            _norm1 = Bn.Load(w, $"{p}.nonlinear1.batchnorm", affine: true),
            _linear1 = WhisperOps.EnsureF32(w[$"{p}.linear1.weight"]),
            _norm2 = Bn.Load(w, $"{p}.nonlinear2.batchnorm", affine: true),
            _localW = WhisperOps.EnsureF32(w[$"{p}.cam_layer.linear_local.weight"]),
            _ctx1W = WhisperOps.EnsureF32(w[$"{p}.cam_layer.linear1.weight"]),
            _ctx1B = WhisperOps.EnsureF32(w[$"{p}.cam_layer.linear1.bias"]),
            _ctx2W = WhisperOps.EnsureF32(w[$"{p}.cam_layer.linear2.weight"]),
            _ctx2B = WhisperOps.EnsureF32(w[$"{p}.cam_layer.linear2.bias"]),
        };

        public Tensor Forward(IBackend backend, Tensor x, int dilation)
        {
            // bn_function: nonlinear1 (BN+ReLU) then 1×1 bottleneck. Operates on a copy (x is reused by concat).
            Tensor h = new(x.Shape, DType.F32);
            long bytes = x.ElementCount * 4;
            Buffer.MemoryCopy((void*)x.DataPointer, (void*)h.DataPointer, bytes, bytes);
            Bn1d(h, _norm1); Relu(backend, h);
            Tensor bn = Conv1d(backend, h, _linear1!, null, 1, 0, 1); h.Dispose();   // [1,128,T]
            Bn1d(bn, _norm2); Relu(backend, bn);

            // CAMLayer: y = linear_local(bn); m = sigmoid(linear2(relu(linear1(mean + seg_pool(bn)))))); return y*m.
            int dilPad = dilation;     // (kernel 3 - 1)//2 * dilation = dilation
            Tensor y = Conv1d(backend, bn, _localW!, null, 1, dilPad, dilation);     // [1,32,T]
            Tensor ctx = ContextPool(bn);                                            // [1,128,T]
            bn.Dispose();
            Tensor c1 = Conv1d(backend, ctx, _ctx1W!, _ctx1B, 1, 0, 1); ctx.Dispose();
            Relu(backend, c1);
            Tensor m = Conv1d(backend, c1, _ctx2W!, _ctx2B, 1, 0, 1); c1.Dispose();
            backend.Sigmoid(m, m);
            MulInPlace(y, m); m.Dispose();
            return y;
        }

        /// <summary>CAM context: <c>x.mean(time) + seg_pool(x)</c>, where seg_pool averages each 100-frame
        /// segment (ceil mode) and broadcasts it back over the frames it covers.</summary>
        private static Tensor ContextPool(Tensor x)
        {
            int C = (int)x.Shape[1], T = (int)x.Shape[2];
            const int seg = 100;
            Tensor o = new(new TensorShape(1, C, T), DType.F32);
            float* p = (float*)x.DataPointer, op = (float*)o.DataPointer;
            for (int c = 0; c < C; c++)
            {
                long baseI = (long)c * T;
                double mean = 0;
                for (int j = 0; j < T; j++) mean += p[baseI + j];
                mean /= T;
                for (int s = 0; s < T; s += seg)
                {
                    int end = Math.Min(s + seg, T);
                    double segSum = 0;
                    for (int j = s; j < end; j++) segSum += p[baseI + j];
                    // ONNX AveragePool (ceil_mode, no pad) divides by the kernel size, not the valid count —
                    // CosyVoice2 deploys the ONNX, so partial windows divide by seg_len too.
                    float segMean = (float)(segSum / seg);
                    for (int j = s; j < end; j++) op[baseI + j] = (float)mean + segMean;
                }
            }
            return o;
        }

        public IEnumerable<Tensor> AllWeights()
        {
            foreach (Tensor t in _norm1.Weights()) yield return t;
            if (_linear1 is not null) yield return _linear1;
            foreach (Tensor t in _norm2.Weights()) yield return t;
            Tensor?[] cam = [_localW, _ctx1W, _ctx1B, _ctx2W, _ctx2B];
            foreach (Tensor? t in cam) if (t is not null) yield return t;
        }

        private static void MulInPlace(Tensor dst, Tensor src)
        {
            float* d = (float*)dst.DataPointer, s = (float*)src.DataPointer;
            for (long i = 0; i < dst.ElementCount; i++) d[i] *= s[i];
        }
    }
}
