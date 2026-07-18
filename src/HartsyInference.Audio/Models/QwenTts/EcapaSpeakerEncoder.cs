using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.QwenTts;

/// <summary>ECAPA-TDNN speaker encoder for Qwen3-TTS x-vector voice cloning — a faithful port of the Base
/// checkpoint's <c>Qwen3TTSSpeakerEncoder</c> (<c>speaker_encoder.*</c>). Takes a 128-bin log-mel
/// <c>[1, 128, T]</c> and returns the <c>enc_dim</c> (2048) x-vector <c>[1, 2048]</c>, which the talker injects
/// directly into the codec stream at the speaker position.
///
/// <para>Topology (NO BatchNorm — the reference TDNN blocks are Conv1d+ReLU only): a TDNN stem
/// (<c>blocks.0</c>, Conv1d k5) → three SE-Res2Net blocks (<c>blocks.1..3</c>: tdnn1 k1 → nested Res2Net
/// (chunk×scale, cascaded k3 TDNNs) → tdnn2 k1 → squeeze-excite → +residual) → multi-layer aggregation (concat
/// the three SE-Res2 outputs = 1536 → <c>mfa</c> Conv1d k1 + ReLU) → attentive statistics pooling
/// (<c>asp.tdnn</c>+ReLU → tanh → <c>asp.conv</c> → softmax → weighted mean‖std) → <c>fc</c> Conv1d 3072→2048.
/// No post-fc projection.</para></summary>
public sealed unsafe class EcapaSpeakerEncoder : IDisposable
{
    private readonly EcapaConfig _cfg;
    private TdnnBlock? _stem;                       // blocks.0
    private readonly SeRes2Block[] _blocks;         // blocks.1..N (SE-Res2Net)
    private Tensor? _mfaW, _mfaB;                    // mfa.conv (1536→1536, k1)
    private Tensor? _aspTdnnW, _aspTdnnB;            // asp.tdnn.conv (4608→128, k1)
    private Tensor? _aspConvW, _aspConvB;            // asp.conv (128→1536, k1)
    private Tensor? _fcW, _fcB;                      // fc (3072→2048, k1)
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
        // Stem = blocks.0 (TDNN Conv1d k5, no BN).
        _stem = new TdnnBlock(_cfg.InputChannels, _cfg.StemChannels, _cfg.StemKernelSize, 1);
        _stem.LoadWeights(w, $"{prefix}.blocks.0");
        // SE-Res2Net blocks = blocks.1..N.
        int inCh = _cfg.StemChannels;
        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i] = new SeRes2Block(inCh, _cfg.Channels[i], _cfg.KernelSizes[i], _cfg.Dilations[i], _cfg.Res2Scale, _cfg.SeBottleneck);
            _blocks[i].LoadWeights(w, $"{prefix}.blocks.{i + 1}");
            inCh = _cfg.Channels[i];
        }
        _mfaW = WhisperOps.EnsureF32(w[$"{prefix}.mfa.conv.weight"]);
        _mfaB = TryGet(w, $"{prefix}.mfa.conv.bias");
        _aspTdnnW = WhisperOps.EnsureF32(w[$"{prefix}.asp.tdnn.conv.weight"]);
        _aspTdnnB = TryGet(w, $"{prefix}.asp.tdnn.conv.bias");
        _aspConvW = WhisperOps.EnsureF32(w[$"{prefix}.asp.conv.weight"]);
        _aspConvB = TryGet(w, $"{prefix}.asp.conv.bias");
        _fcW = WhisperOps.EnsureF32(w[$"{prefix}.fc.weight"]);
        _fcB = TryGet(w, $"{prefix}.fc.bias");
    }

    /// <summary>Runs the encoder over a log-mel <c>[1, nMels, T]</c> and returns the x-vector <c>[1, EmbeddingDim]</c>.</summary>
    public Tensor Encode(IBackend backend, Tensor mel)
    {
        int t = (int)mel.Shape[2];
        Tensor stemOut = _stem!.Forward(backend, mel, t);

        // Three SE-Res2 blocks; their outputs are aggregated (MFA concats blocks[1:] = the SE-Res2 outputs).
        Tensor[] outs = new Tensor[_blocks.Length];
        Tensor cur = stemOut;
        for (int i = 0; i < _blocks.Length; i++)
        {
            outs[i] = _blocks[i].Forward(backend, cur, t);
            cur = outs[i];
        }
        Tensor concat = ConcatChannels(outs, t);
        stemOut.Dispose();
        for (int i = 0; i < outs.Length; i++) outs[i].Dispose();

        // MFA: Conv1d(1536→1536, k1) + ReLU.
        int mfaCh = _cfg.MfaChannels;
        Tensor mfa = new(new TensorShape(1, mfaCh, t), DType.F32);
        backend.Conv1d(mfa, concat, _mfaW!, _mfaB, 1, 0, 0, 1, 1);
        concat.Dispose();
        Relu(mfa);

        Tensor pooled = AttentiveStatPooling(backend, mfa, mfaCh, t);   // [1, 1, 2*mfaCh]
        mfa.Dispose();

        // fc: Conv1d(3072→2048, k1). Output IS the x-vector (no bn/proj).
        Tensor emb = WhisperOps.ProjectLinear(backend, pooled, _fcW!, _fcB, 1, 1, 2 * mfaCh, _cfg.EmbeddingDim);
        pooled.Dispose();

        Tensor vec = new(new TensorShape(1, _cfg.EmbeddingDim), DType.F32);
        Buffer.MemoryCopy((void*)emb.DataPointer, (void*)vec.DataPointer, (long)_cfg.EmbeddingDim * 4, (long)_cfg.EmbeddingDim * 4);
        emb.Dispose();
        return vec;
    }

    /// <summary>Attentive statistics pooling: build [frame, globalMean, globalStd] → asp.tdnn (Conv1d k1) + ReLU →
    /// tanh → asp.conv (Conv1d k1) → softmax over time → weighted mean and std, concatenated. Returns
    /// <c>[1, 1, 2*channels]</c> (rank-3 so the downstream fc reads 2*channels as the input dim).</summary>
    private Tensor AttentiveStatPooling(IBackend backend, Tensor x, int channels, int t)
    {
        float* xp = (float*)x.DataPointer;
        float[] gMean = new float[channels];
        float[] gStd = new float[channels];
        for (int c = 0; c < channels; c++)
        {
            double m = 0; for (int s = 0; s < t; s++) m += xp[(long)c * t + s]; m /= t;
            double v = 0; for (int s = 0; s < t; s++) { double d = xp[(long)c * t + s] - m; v += d * d; } v /= t;
            gMean[c] = (float)m; gStd[c] = (float)Math.Sqrt(v + 1e-12);
        }

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

        // asp.tdnn (Conv1d ctxCh→attDim, k1) + ReLU, then tanh, then asp.conv (attDim→channels, k1).
        int attDim = _cfg.AttentionChannels;
        Tensor a1 = new(new TensorShape(1, attDim, t), DType.F32);
        backend.Conv1d(a1, ctx, _aspTdnnW!, _aspTdnnB, 1, 0, 0, 1, 1);
        ctx.Dispose();
        Relu(a1);
        backend.Tanh(a1, a1);
        Tensor scores = new(new TensorShape(1, channels, t), DType.F32);
        backend.Conv1d(scores, a1, _aspConvW!, _aspConvB, 1, 0, 0, 1, 1);
        a1.Dispose();
        SoftmaxOverTime(scores, channels, t);

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

    private static void Relu(Tensor x)
    {
        float* p = (float*)x.DataPointer;
        long n = x.ElementCount;
        for (long i = 0; i < n; i++) if (p[i] < 0) p[i] = 0f;
    }

    /// <summary>Concatenates same-time-length <c>[1, C_i, T]</c> tensors along the channel axis.</summary>
    private static Tensor ConcatChannels(Tensor[] parts, int t)
    {
        int total = 0;
        for (int i = 0; i < parts.Length; i++) total += (int)parts[i].Shape[1];
        Tensor outT = new(new TensorShape(1, total, t), DType.F32);
        float* op = (float*)outT.DataPointer;
        long off = 0;
        for (int i = 0; i < parts.Length; i++)
        {
            long bytes = parts[i].ElementCount * 4;
            Buffer.MemoryCopy((void*)parts[i].DataPointer, op + off, bytes, bytes);
            off += parts[i].ElementCount;
        }
        return outT;
    }

    private static Tensor? TryGet(IReadOnlyDictionary<string, Tensor> w, string k)
        => w.TryGetValue(k, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_stem is not null) foreach (Tensor t in _stem.EnumerateWeights()) yield return t;
        foreach (SeRes2Block b in _blocks) if (b is not null) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        Tensor?[] own = [_mfaW, _mfaB, _aspTdnnW, _aspTdnnB, _aspConvW, _aspConvB, _fcW, _fcB];
        foreach (Tensor? t in own) if (t is not null) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    /// <summary>TDNN block: dilated Conv1d ("same" symmetric pad) → ReLU. No BatchNorm (Qwen3 variant).</summary>
    private sealed class TdnnBlock
    {
        private readonly int _outCh, _kernel, _dilation;
        private Tensor? _convW, _convB;

        public TdnnBlock(int inCh, int outCh, int kernel, int dilation) { _outCh = outCh; _kernel = kernel; _dilation = dilation; }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _convW = WhisperOps.EnsureF32(w[$"{p}.conv.weight"]);
            _convB = TryGet(w, $"{p}.conv.bias");
        }

        public Tensor Forward(IBackend backend, Tensor x, int t)
        {
            int pad = (_kernel - 1) * _dilation / 2;   // "same"
            Tensor h = new(new TensorShape(1, _outCh, t), DType.F32);
            backend.Conv1d(h, x, _convW!, _convB, 1, pad, pad, _dilation, 1);
            Relu(h);
            return h;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            if (_convW is not null) yield return _convW;
            if (_convB is not null) yield return _convB;
        }
    }

    /// <summary>SE-Res2Net block: tdnn1 (k1) → nested Res2Net (chunk into <c>scale</c>, cascaded k-TDNNs) →
    /// tdnn2 (k1) → squeeze-excite (time-mean → conv1 → ReLU → conv2 → sigmoid → channel gate) → + residual.</summary>
    private sealed class SeRes2Block
    {
        private readonly int _outCh, _kernel, _dilation, _scale, _seBottleneck;
        private Tensor? _t1W, _t1B, _t2W, _t2B;                    // tdnn1 / tdnn2 (k1)
        private readonly Tensor?[] _resW, _resB;                    // res2net_block.blocks.{m}.conv
        private Tensor? _seC1W, _seC1B, _seC2W, _seC2B;            // se_block.conv1 / conv2

        public SeRes2Block(int inCh, int outCh, int kernel, int dilation, int scale, int seBottleneck)
        {
            _outCh = outCh; _kernel = kernel; _dilation = dilation; _scale = scale; _seBottleneck = seBottleneck;
            _resW = new Tensor?[scale - 1];
            _resB = new Tensor?[scale - 1];
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
        {
            _t1W = WhisperOps.EnsureF32(w[$"{p}.tdnn1.conv.weight"]); _t1B = TryGet(w, $"{p}.tdnn1.conv.bias");
            for (int m = 0; m < _scale - 1; m++)
            {
                _resW[m] = WhisperOps.EnsureF32(w[$"{p}.res2net_block.blocks.{m}.conv.weight"]);
                _resB[m] = TryGet(w, $"{p}.res2net_block.blocks.{m}.conv.bias");
            }
            _t2W = WhisperOps.EnsureF32(w[$"{p}.tdnn2.conv.weight"]); _t2B = TryGet(w, $"{p}.tdnn2.conv.bias");
            _seC1W = WhisperOps.EnsureF32(w[$"{p}.se_block.conv1.weight"]); _seC1B = TryGet(w, $"{p}.se_block.conv1.bias");
            _seC2W = WhisperOps.EnsureF32(w[$"{p}.se_block.conv2.weight"]); _seC2B = TryGet(w, $"{p}.se_block.conv2.bias");
        }

        public Tensor Forward(IBackend backend, Tensor x, int t)
        {
            // tdnn1 (k1 conv + relu).
            Tensor h = new(new TensorShape(1, _outCh, t), DType.F32);
            backend.Conv1d(h, x, _t1W!, _t1B, 1, 0, 0, 1, 1);
            Relu(h);

            // Res2Net: chunk h into `scale` parts (each _outCh/scale channels); cascade the k-TDNN blocks.
            Tensor r = Res2Net(backend, h, t);
            h.Dispose();

            // tdnn2 (k1 conv + relu).
            Tensor c = new(new TensorShape(1, _outCh, t), DType.F32);
            backend.Conv1d(c, r, _t2W!, _t2B, 1, 0, 0, 1, 1);
            r.Dispose();
            Relu(c);

            // Squeeze-excite gate.
            SqueezeExcite(backend, c, _outCh, t);

            // Residual (in==out).
            float* cp = (float*)c.DataPointer;
            float* xp = (float*)x.DataPointer;
            long n = c.ElementCount;
            for (long i = 0; i < n; i++) cp[i] += xp[i];
            return c;
        }

        /// <summary>Res2Net: split channels into <c>scale</c> chunks; chunk 0 passes through, chunk 1 → block[0],
        /// chunk i (≥2) → block[i-1] of (chunk i + previous block output). Concatenate.</summary>
        private Tensor Res2Net(IBackend backend, Tensor x, int t)
        {
            int hidden = _outCh / _scale;
            int pad = (_kernel - 1) * _dilation / 2;
            float* xp = (float*)x.DataPointer;
            Tensor outT = new(new TensorShape(1, _outCh, t), DType.F32);
            float* op = (float*)outT.DataPointer;

            float[] prev = new float[(long)hidden * t];       // previous block's output part
            float[] inbuf = new float[(long)hidden * t];      // chunk (+ prev) input to a block
            for (int i = 0; i < _scale; i++)
            {
                int cbase = i * hidden;
                if (i == 0)
                {
                    // pass-through
                    for (int c = 0; c < hidden; c++)
                        for (int s = 0; s < t; s++) op[(long)(cbase + c) * t + s] = xp[(long)(cbase + c) * t + s];
                    continue;
                }
                // input part = chunk i, plus previous output for i>=2
                for (int c = 0; c < hidden; c++)
                    for (int s = 0; s < t; s++)
                    {
                        float v = xp[(long)(cbase + c) * t + s];
                        if (i >= 2) v += prev[(long)c * t + s];
                        inbuf[(long)c * t + s] = v;
                    }
                Tensor inp = new(new TensorShape(1, hidden, t), DType.F32);
                fixed (float* ib = inbuf) Buffer.MemoryCopy(ib, (void*)inp.DataPointer, (long)hidden * t * 4, (long)hidden * t * 4);
                Tensor bo = new(new TensorShape(1, hidden, t), DType.F32);
                backend.Conv1d(bo, inp, _resW[i - 1]!, _resB[i - 1], 1, pad, pad, _dilation, 1);
                inp.Dispose();
                Relu(bo);
                float* bop = (float*)bo.DataPointer;
                for (int c = 0; c < hidden; c++)
                    for (int s = 0; s < t; s++)
                    {
                        float v = bop[(long)c * t + s];
                        op[(long)(cbase + c) * t + s] = v;
                        prev[(long)c * t + s] = v;
                    }
                bo.Dispose();
            }
            return outT;
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
            Tensor s1 = WhisperOps.ProjectLinear(backend, avg, _seC1W!, _seC1B, 1, 1, channels, _seBottleneck);
            avg.Dispose();
            float* s1p = (float*)s1.DataPointer;
            for (int i = 0; i < _seBottleneck; i++) if (s1p[i] < 0) s1p[i] = 0f;
            Tensor s2 = WhisperOps.ProjectLinear(backend, s1, _seC2W!, _seC2B, 1, 1, _seBottleneck, channels);
            s1.Dispose();
            float* s2p = (float*)s2.DataPointer;
            for (int c = 0; c < channels; c++)
            {
                float g = 1f / (1f + MathF.Exp(-s2p[c]));
                for (int s = 0; s < t; s++) xp[(long)c * t + s] *= g;
            }
            s2.Dispose();
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] a = [_t1W, _t1B, _t2W, _t2B, _seC1W, _seC1B, _seC2W, _seC2B];
            foreach (Tensor? t in a) if (t is not null) yield return t;
            for (int m = 0; m < _resW.Length; m++) { if (_resW[m] is not null) yield return _resW[m]!; if (_resB[m] is not null) yield return _resB[m]!; }
        }
    }
}
