using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.QwenTts;

/// <summary>ECAPA-TDNN speaker encoder for Qwen3-TTS x-vector voice cloning. Takes a log-mel feature sequence
/// <c>[1, nMels, T]</c> (the caller front-end produces it via <c>MelSpectrogramExtractor</c>) and produces a
/// fixed 192-d speaker embedding, then projects it to the talker's speaker-conditioning width.
///
/// <para>Pipeline: a TDNN stem (Conv1d k5 + ReLU + BatchNorm) → a stack of SE-Res2Blocks (dilated Res2 conv +
/// squeeze-excite) whose per-block outputs are concatenated (multi-layer feature aggregation) → a 1×1
/// aggregation conv → attentive statistics pooling (context-dependent attention over time → weighted mean and
/// std) → a final linear + BatchNorm to 192-d → an output projection to the conditioning dim.</para></summary>
public sealed unsafe class EcapaSpeakerEncoder : IDisposable
{
    private readonly EcapaConfig _cfg;
    private TdnnBlock? _stem;
    private readonly SeRes2Block[] _blocks;
    private Tensor? _aggW, _aggB;                 // 1×1 conv over concatenated block outputs
    private Tensor? _attW1, _attB1, _attW2, _attB2;   // attentive stat pooling MLP (context conv)
    private Tensor? _fcW, _fcB;                    // → embedding (192)
    private Tensor? _bnW, _bnB, _bnMean, _bnVar;   // final BatchNorm over 192
    private Tensor? _projW, _projB;                // 192 → conditioning dim
    private int _disposed;

    public EcapaConfig Config => _cfg;
    public int EmbeddingDim => _cfg.EmbeddingDim;

    public EcapaSpeakerEncoder(EcapaConfig cfg)
    {
        _cfg = cfg;
        _blocks = new SeRes2Block[cfg.Channels.Count];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "speaker_encoder")
    {
        _stem = new TdnnBlock(_cfg.InputChannels, _cfg.StemChannels, 5, 1);
        _stem.LoadWeights(w, $"{prefix}.tdnn");
        int inCh = _cfg.StemChannels;
        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i] = new SeRes2Block(inCh, _cfg.Channels[i], _cfg.KernelSizes[i], _cfg.Dilations[i], _cfg.Res2Scale, _cfg.SeBottleneck);
            _blocks[i].LoadWeights(w, $"{prefix}.blocks.{i}");
            inCh = _cfg.Channels[i];
        }
        _aggW = WhisperOps.EnsureF32(w[$"{prefix}.mfa.weight"]);
        _aggB = TryGet(w, $"{prefix}.mfa.bias");
        _attW1 = WhisperOps.EnsureF32(w[$"{prefix}.asp.tdnn.weight"]);
        _attB1 = TryGet(w, $"{prefix}.asp.tdnn.bias");
        _attW2 = WhisperOps.EnsureF32(w[$"{prefix}.asp.conv.weight"]);
        _attB2 = TryGet(w, $"{prefix}.asp.conv.bias");
        _fcW = WhisperOps.EnsureF32(w[$"{prefix}.fc.weight"]);
        _fcB = TryGet(w, $"{prefix}.fc.bias");
        _bnW = WhisperOps.EnsureF32(w[$"{prefix}.bn.weight"]);
        _bnB = WhisperOps.EnsureF32(w[$"{prefix}.bn.bias"]);
        _bnMean = WhisperOps.EnsureF32(w[$"{prefix}.bn.running_mean"]);
        _bnVar = WhisperOps.EnsureF32(w[$"{prefix}.bn.running_var"]);
        _projW = WhisperOps.EnsureF32(w[$"{prefix}.proj.weight"]);
        _projB = TryGet(w, $"{prefix}.proj.bias");
    }

    /// <summary>Runs the encoder over a log-mel feature sequence <c>[1, nMels, T]</c> and returns the projected
    /// speaker conditioning vector <c>[1, condDim]</c>.</summary>
    public Tensor Encode(IBackend backend, Tensor mel)
    {
        int t = (int)mel.Shape[2];
        Tensor h = _stem!.Forward(backend, mel, t);

        // Multi-layer feature aggregation: concatenate each block's output along channels.
        Tensor[] outs = new Tensor[_blocks.Length];
        Tensor cur = h;
        for (int i = 0; i < _blocks.Length; i++)
        {
            outs[i] = _blocks[i].Forward(backend, cur, t);
            cur = outs[i];
        }
        Tensor concat = ConcatChannels(outs, t);
        h.Dispose();
        for (int i = 0; i < outs.Length; i++) outs[i].Dispose();

        // 1×1 aggregation conv + ReLU.
        int aggCh = (int)_aggW!.Shape[0];
        Tensor agg = new(new TensorShape(1, aggCh, t), DType.F32);
        backend.Conv1d(agg, concat, _aggW!, _aggB, 1, 0, 0, 1, 1);
        concat.Dispose();
        Relu(agg);

        Tensor pooled = AttentiveStatPooling(backend, agg, aggCh, t);   // [1, 1, 2*aggCh]
        agg.Dispose();

        // FC → embedding + final BatchNorm.
        Tensor emb = WhisperOps.ProjectLinear(backend, pooled, _fcW!, _fcB, 1, 1, 2 * aggCh, _cfg.EmbeddingDim);
        pooled.Dispose();
        BatchNorm1d(emb, _cfg.EmbeddingDim);

        Tensor projected = WhisperOps.ProjectLinear(backend, emb, _projW!, _projB, 1, 1, _cfg.EmbeddingDim, _cfg.ConditioningDim);
        emb.Dispose();

        // ProjectLinear emits [1, 1, condDim]; the public contract is the rank-2 conditioning vector [1, condDim].
        Tensor cond = new(new TensorShape(1, _cfg.ConditioningDim), DType.F32);
        long condBytes = (long)_cfg.ConditioningDim * 4;
        Buffer.MemoryCopy((void*)projected.DataPointer, (void*)cond.DataPointer, condBytes, condBytes);
        projected.Dispose();
        return cond;
    }

    /// <summary>Attentive statistics pooling: a context-dependent attention (concat of frame, global mean and
    /// std → tdnn → tanh → conv → softmax over time) yields per-channel weights, then computes the weighted mean
    /// and weighted std and concatenates them. Returns <c>[1, 1, 2*channels]</c> (rank-3 so the downstream FC
    /// ProjectLinear reads <c>2*channels</c> as the input dim, not the matmul row count).</summary>
    private Tensor AttentiveStatPooling(IBackend backend, Tensor x, int channels, int t)
    {
        // Global mean / std over time for the context.
        float* xp = (float*)x.DataPointer;
        float[] gMean = new float[channels];
        float[] gStd = new float[channels];
        for (int c = 0; c < channels; c++)
        {
            double m = 0; for (int s = 0; s < t; s++) m += xp[(long)c * t + s]; m /= t;
            double v = 0; for (int s = 0; s < t; s++) { double d = xp[(long)c * t + s] - m; v += d * d; } v /= t;
            gMean[c] = (float)m; gStd[c] = (float)Math.Sqrt(v + 1e-12);
        }

        // Build attention input [1, 3*channels, T] = [frame, gMean broadcast, gStd broadcast].
        int ctxCh = 3 * channels;
        Tensor ctx = new(new TensorShape(1, ctxCh, t), DType.F32);
        float* cp = (float*)ctx.DataPointer;
        for (int c = 0; c < channels; c++)
            for (int s = 0; s < t; s++)
            {
                cp[(long)c * t + s] = xp[(long)c * t + s];
                cp[(long)(channels + c) * t + s] = gMean[c];
                cp[(long)(2 * channels + c) * t + s] = gStd[c];
            }

        // tdnn (1×1 conv ctxCh→attDim) + tanh, then conv (attDim→channels), then softmax over time.
        int attDim = (int)_attW1!.Shape[0];
        Tensor a1 = new(new TensorShape(1, attDim, t), DType.F32);
        backend.Conv1d(a1, ctx, _attW1!, _attB1, 1, 0, 0, 1, 1);
        ctx.Dispose();
        backend.Tanh(a1, a1);
        Tensor scores = new(new TensorShape(1, channels, t), DType.F32);
        backend.Conv1d(scores, a1, _attW2!, _attB2, 1, 0, 0, 1, 1);
        a1.Dispose();
        SoftmaxOverTime(scores, channels, t);

        // Weighted mean and std per channel. Laid out [1, 1, 2*channels] so the downstream FC ProjectLinear
        // reads it as batch=1, seqLen=1, inDim=2*channels (a rank-2 [1, 2C] tensor would make BatchedMatMul
        // treat 2C as the M dimension and write 2C rows into a 1-row output buffer — a heap overflow).
        float* sp = (float*)scores.DataPointer;
        Tensor pooled = new(new TensorShape(1, 1, 2 * channels), DType.F32);
        float* pp = (float*)pooled.DataPointer;
        for (int c = 0; c < channels; c++)
        {
            double wm = 0; for (int s = 0; s < t; s++) wm += sp[(long)c * t + s] * xp[(long)c * t + s];
            double wv = 0; for (int s = 0; s < t; s++) { double d = xp[(long)c * t + s]; wv += sp[(long)c * t + s] * d * d; }
            wv -= wm * wm;
            pp[c] = (float)wm;
            pp[channels + c] = (float)Math.Sqrt(Math.Max(wv, 0) + 1e-12);
        }
        scores.Dispose();
        return pooled;
    }

    private static void SoftmaxOverTime(Tensor scores, int channels, int t)
    {
        float* sp = (float*)scores.DataPointer;
        for (int c = 0; c < channels; c++)
        {
            float max = float.NegativeInfinity;
            for (int s = 0; s < t; s++) { float v = sp[(long)c * t + s]; if (v > max) max = v; }
            double sum = 0;
            for (int s = 0; s < t; s++) { float e = MathF.Exp(sp[(long)c * t + s] - max); sp[(long)c * t + s] = e; sum += e; }
            float inv = (float)(1.0 / sum);
            for (int s = 0; s < t; s++) sp[(long)c * t + s] *= inv;
        }
    }

    private void BatchNorm1d(Tensor x, int dim)
    {
        float* xp = (float*)x.DataPointer;
        float* w = (float*)_bnW!.DataPointer;
        float* b = (float*)_bnB!.DataPointer;
        float* mean = (float*)_bnMean!.DataPointer;
        float* var = (float*)_bnVar!.DataPointer;
        for (int d = 0; d < dim; d++)
            xp[d] = (xp[d] - mean[d]) / MathF.Sqrt(var[d] + 1e-5f) * w[d] + b[d];
    }

    private static Tensor ConcatChannels(Tensor[] parts, int t)
    {
        int total = 0;
        for (int i = 0; i < parts.Length; i++) total += (int)parts[i].Shape[1];
        Tensor outT = new(new TensorShape(1, total, t), DType.F32);
        float* op = (float*)outT.DataPointer;
        long off = 0;
        for (int i = 0; i < parts.Length; i++)
        {
            int c = (int)parts[i].Shape[1];
            long bytes = (long)c * t * 4;
            Buffer.MemoryCopy((void*)parts[i].DataPointer, op + off, bytes, bytes);
            off += (long)c * t;
        }
        return outT;
    }

    private static void Relu(Tensor x)
    {
        float* p = (float*)x.DataPointer;
        long n = x.ElementCount;
        for (long i = 0; i < n; i++) if (p[i] < 0) p[i] = 0f;
    }

    private static Tensor? TryGet(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_stem is not null) foreach (Tensor t in _stem.EnumerateWeights()) yield return t;
        foreach (SeRes2Block b in _blocks) if (b is not null) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        Tensor?[] tail = [_aggW, _aggB, _attW1, _attB1, _attW2, _attB2, _fcW, _fcB, _bnW, _bnB, _bnMean, _bnVar, _projW, _projB];
        foreach (Tensor? t in tail) if (t is not null) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    /// <summary>TDNN block: dilated Conv1d (symmetric pad) → ReLU → BatchNorm1d.</summary>
    private sealed class TdnnBlock
    {
        private readonly int _inCh, _outCh, _kernel, _dilation;
        private Tensor? _convW, _convB, _bnW, _bnB, _bnMean, _bnVar;

        public TdnnBlock(int inCh, int outCh, int kernel, int dilation) { _inCh = inCh; _outCh = outCh; _kernel = kernel; _dilation = dilation; }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _convW = WhisperOps.EnsureF32(w[$"{p}.conv.weight"]);
            _convB = TryGet(w, $"{p}.conv.bias");
            _bnW = WhisperOps.EnsureF32(w[$"{p}.bn.weight"]);
            _bnB = WhisperOps.EnsureF32(w[$"{p}.bn.bias"]);
            _bnMean = WhisperOps.EnsureF32(w[$"{p}.bn.running_mean"]);
            _bnVar = WhisperOps.EnsureF32(w[$"{p}.bn.running_var"]);
        }

        public Tensor Forward(IBackend backend, Tensor x, int t)
        {
            int pad = (_kernel - 1) * _dilation / 2;
            Tensor h = new(new TensorShape(1, _outCh, t), DType.F32);
            backend.Conv1d(h, x, _convW!, _convB, 1, pad, pad, _dilation, 1);
            float* p = (float*)h.DataPointer;
            long n = h.ElementCount;
            for (long i = 0; i < n; i++) if (p[i] < 0) p[i] = 0f;
            ApplyBatchNorm(h, _outCh, t, _bnW!, _bnB!, _bnMean!, _bnVar!);
            return h;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] a = [_convW, _convB, _bnW, _bnB, _bnMean, _bnVar];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }

    /// <summary>SE-Res2Block: 1×1 conv → ReLU+BN → Res2 dilated conv (multi-scale within channels) → ReLU+BN →
    /// 1×1 conv → ReLU+BN → squeeze-excite channel gate → + residual.</summary>
    private sealed class SeRes2Block
    {
        private readonly int _inCh, _outCh, _kernel, _dilation, _scale, _seBottleneck;
        private Tensor? _c1W, _c1B, _c1bnW, _c1bnB, _c1bnM, _c1bnV;
        private Tensor? _resW, _resB, _resbnW, _resbnB, _resbnM, _resbnV;
        private Tensor? _c3W, _c3B, _c3bnW, _c3bnB, _c3bnM, _c3bnV;
        private Tensor? _seW1, _seB1, _seW2, _seB2;

        public SeRes2Block(int inCh, int outCh, int kernel, int dilation, int scale, int seBottleneck)
        {
            _inCh = inCh; _outCh = outCh; _kernel = kernel; _dilation = dilation; _scale = scale; _seBottleneck = seBottleneck;
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _c1W = WhisperOps.EnsureF32(w[$"{p}.conv1.weight"]); _c1B = TryGet(w, $"{p}.conv1.bias");
            _c1bnW = WhisperOps.EnsureF32(w[$"{p}.bn1.weight"]); _c1bnB = WhisperOps.EnsureF32(w[$"{p}.bn1.bias"]);
            _c1bnM = WhisperOps.EnsureF32(w[$"{p}.bn1.running_mean"]); _c1bnV = WhisperOps.EnsureF32(w[$"{p}.bn1.running_var"]);
            _resW = WhisperOps.EnsureF32(w[$"{p}.res2.weight"]); _resB = TryGet(w, $"{p}.res2.bias");
            _resbnW = WhisperOps.EnsureF32(w[$"{p}.bn2.weight"]); _resbnB = WhisperOps.EnsureF32(w[$"{p}.bn2.bias"]);
            _resbnM = WhisperOps.EnsureF32(w[$"{p}.bn2.running_mean"]); _resbnV = WhisperOps.EnsureF32(w[$"{p}.bn2.running_var"]);
            _c3W = WhisperOps.EnsureF32(w[$"{p}.conv3.weight"]); _c3B = TryGet(w, $"{p}.conv3.bias");
            _c3bnW = WhisperOps.EnsureF32(w[$"{p}.bn3.weight"]); _c3bnB = WhisperOps.EnsureF32(w[$"{p}.bn3.bias"]);
            _c3bnM = WhisperOps.EnsureF32(w[$"{p}.bn3.running_mean"]); _c3bnV = WhisperOps.EnsureF32(w[$"{p}.bn3.running_var"]);
            _seW1 = WhisperOps.EnsureF32(w[$"{p}.se.fc1.weight"]); _seB1 = TryGet(w, $"{p}.se.fc1.bias");
            _seW2 = WhisperOps.EnsureF32(w[$"{p}.se.fc2.weight"]); _seB2 = TryGet(w, $"{p}.se.fc2.bias");
        }

        public Tensor Forward(IBackend backend, Tensor x, int t)
        {
            // conv1 1×1 → ReLU+BN.
            Tensor h = new(new TensorShape(1, _outCh, t), DType.F32);
            backend.Conv1d(h, x, _c1W!, _c1B, 1, 0, 0, 1, 1);
            ReluBn(h, _outCh, t, _c1bnW!, _c1bnB!, _c1bnM!, _c1bnV!);

            // Res2 dilated conv (grouped multi-scale within channels) → ReLU+BN.
            int pad = (_kernel - 1) * _dilation / 2;
            Tensor r = new(new TensorShape(1, _outCh, t), DType.F32);
            backend.Conv1d(r, h, _resW!, _resB, 1, pad, pad, _dilation, _scale);
            h.Dispose();
            ReluBn(r, _outCh, t, _resbnW!, _resbnB!, _resbnM!, _resbnV!);

            // conv3 1×1 → ReLU+BN.
            Tensor c3 = new(new TensorShape(1, _outCh, t), DType.F32);
            backend.Conv1d(c3, r, _c3W!, _c3B, 1, 0, 0, 1, 1);
            r.Dispose();
            ReluBn(c3, _outCh, t, _c3bnW!, _c3bnB!, _c3bnM!, _c3bnV!);

            // Squeeze-excite: global avg pool → fc1 ReLU → fc2 sigmoid → channel gate.
            SqueezeExcite(backend, c3, _outCh, t);

            // Residual (channel-matched when in==out; otherwise the 1×1 conv1 already mapped, add x only if shapes match).
            if (_inCh == _outCh)
            {
                float* cp = (float*)c3.DataPointer;
                float* xp = (float*)x.DataPointer;
                long n = c3.ElementCount;
                for (long i = 0; i < n; i++) cp[i] += xp[i];
            }
            return c3;
        }

        private void SqueezeExcite(IBackend backend, Tensor x, int channels, int t)
        {
            float* xp = (float*)x.DataPointer;
            Tensor avg = new(new TensorShape(1, 1, channels), DType.F32);
            float* ap = (float*)avg.DataPointer;
            for (int c = 0; c < channels; c++)
            {
                double m = 0; for (int s = 0; s < t; s++) m += xp[(long)c * t + s];
                ap[c] = (float)(m / t);
            }
            Tensor s1 = WhisperOps.ProjectLinear(backend, avg, _seW1!, _seB1, 1, 1, channels, _seBottleneck);
            avg.Dispose();
            float* s1p = (float*)s1.DataPointer;
            for (int i = 0; i < _seBottleneck; i++) if (s1p[i] < 0) s1p[i] = 0f;
            Tensor s2 = WhisperOps.ProjectLinear(backend, s1, _seW2!, _seB2, 1, 1, _seBottleneck, channels);
            s1.Dispose();
            float* s2p = (float*)s2.DataPointer;
            for (int c = 0; c < channels; c++)
            {
                float g = 1f / (1f + MathF.Exp(-s2p[c]));
                for (int s = 0; s < t; s++) xp[(long)c * t + s] *= g;
            }
            s2.Dispose();
        }

        private static void ReluBn(Tensor x, int c, int t, Tensor bnW, Tensor bnB, Tensor bnM, Tensor bnV)
        {
            float* p = (float*)x.DataPointer;
            long n = x.ElementCount;
            for (long i = 0; i < n; i++) if (p[i] < 0) p[i] = 0f;
            ApplyBatchNorm(x, c, t, bnW, bnB, bnM, bnV);
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] a =
            [
                _c1W, _c1B, _c1bnW, _c1bnB, _c1bnM, _c1bnV,
                _resW, _resB, _resbnW, _resbnB, _resbnM, _resbnV,
                _c3W, _c3B, _c3bnW, _c3bnB, _c3bnM, _c3bnV,
                _seW1, _seB1, _seW2, _seB2,
            ];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }

    /// <summary>Applies inference-mode BatchNorm1d over a <c>[1, C, T]</c> tensor in place using running stats.</summary>
    private static void ApplyBatchNorm(Tensor x, int c, int t, Tensor bnW, Tensor bnB, Tensor bnMean, Tensor bnVar)
    {
        float* xp = (float*)x.DataPointer;
        float* w = (float*)bnW.DataPointer;
        float* b = (float*)bnB.DataPointer;
        float* mean = (float*)bnMean.DataPointer;
        float* var = (float*)bnVar.DataPointer;
        for (int ci = 0; ci < c; ci++)
        {
            float scale = w[ci] / MathF.Sqrt(var[ci] + 1e-5f);
            float shift = b[ci] - mean[ci] * scale;
            for (int s = 0; s < t; s++)
            {
                long idx = (long)ci * t + s;
                xp[idx] = xp[idx] * scale + shift;
            }
        }
    }
}
