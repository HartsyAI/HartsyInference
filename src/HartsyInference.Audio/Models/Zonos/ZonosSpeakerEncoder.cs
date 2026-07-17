using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Zonos;

/// <summary>Zonos speaker encoder — the separate-download <c>ResNet293_based</c> model
/// (<c>Zyphra/Zonos-v0.1-speaker-embedding</c>). A 16 kHz logFbank front-end (torchaudio MelSpectrogram +
/// <c>log(mel + 1e-6)</c> + per-mel time-mean subtraction) feeds a ResNet293 trunk of SimAM BasicBlocks
/// (stages <c>[10,20,64,3]</c>, widths <c>[64,128,256,512]</c>) → Attentive Statistics Pooling (ASP) over the
/// folded (channel·freq) axis → a linear bottleneck to 256-d → an LDA linear projection to the 128-d speaker
/// embedding consumed by the Zonos prefix conditioner.
///
/// <para>Every BatchNorm2d is folded into the preceding (bias-free) convolution at load — <c>W' = W·γ/√(σ²+ε)</c>,
/// <c>b' = β − μ·γ/√(σ²+ε)</c> — matching the engine's <c>ConvBnSilu</c>/<c>RmbgBlocks</c> precedent, so the conv
/// trunk runs on the backend with no separate norm pass. SimAM, the ASP softmax/statistics, and the ASP's
/// standalone BatchNorm1d run host-side (the encoder runs once per generation, off the token hot path).</para></summary>
public sealed unsafe class ZonosSpeakerEncoder : IDisposable
{
    private readonly ZonosSpeakerConfig _cfg;
    private readonly MelSpectrogramExtractor _mel;
    private readonly Stage[] _stages;

    private ConvBn? _stem;                       // Conv2d(1→64,k3,s1,p1) + folded bn
    private Tensor? _aspAttn0W, _aspAttn0B;      // Conv1d(5120→128,k1)
    private Tensor? _aspBnScale, _aspBnShift;    // folded BatchNorm1d(128)
    private Tensor? _aspAttn3W, _aspAttn3B;      // Conv1d(128→5120,k1)
    private Tensor? _bottleW, _bottleB;          // Linear(10240→256)
    private Tensor? _ldaW, _ldaB;                // Linear(256→128)
    private int _disposed;

    public ZonosSpeakerConfig Config => _cfg;

    public ZonosSpeakerEncoder(ZonosSpeakerConfig cfg)
    {
        _cfg = cfg;
        _mel = new MelSpectrogramExtractor(MelSpectrogramExtractor.Zonos16kConfig());
        _stages = new Stage[cfg.StageBlocks.Count];
    }

    /// <summary>Loads the ResNet293 weights (<paramref name="w"/>, keys under <c>front.*</c>/<c>pooling.*</c>/
    /// <c>bottleneck.*</c>) and the separate LDA head (<paramref name="lda"/>, keys <c>weight</c>/<c>bias</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, IReadOnlyDictionary<string, Tensor> lda)
    {
        _stem = FoldConvBn(w, "front.conv1", "front.bn1");

        int inC = _cfg.BaseWidth;
        for (int s = 0; s < _stages.Length; s++)
        {
            int outC = _cfg.StageWidths[s];
            int stageStride = s == 0 ? 1 : 2;
            SimAmBlock[] blocks = new SimAmBlock[_cfg.StageBlocks[s]];
            for (int b = 0; b < blocks.Length; b++)
            {
                string bp = $"front.layer{s + 1}.{b}";
                int blockStride = b == 0 ? stageStride : 1;
                int blockIn = b == 0 ? inC : outC;
                bool hasDown = blockStride != 1 || blockIn != outC;
                blocks[b] = new SimAmBlock
                {
                    Conv1 = FoldConvBn(w, $"{bp}.conv1", $"{bp}.bn1"),
                    Conv2 = FoldConvBn(w, $"{bp}.conv2", $"{bp}.bn2"),
                    Down = hasDown ? FoldConvBn(w, $"{bp}.downsample.0", $"{bp}.downsample.1") : null,
                    Stride = blockStride,
                    OutChannels = outC,
                };
            }
            _stages[s] = new Stage { Blocks = blocks };
            inC = outC;
        }

        // ASP: Conv1d(5120→128) → ReLU → BatchNorm1d(128) → Conv1d(128→5120) → Softmax(time).
        _aspAttn0W = WhisperOps.EnsureF32(w["pooling.attention.0.weight"]);
        _aspAttn0B = WhisperOps.EnsureF32(w["pooling.attention.0.bias"]);
        FoldBatchNorm1d(w, "pooling.attention.2", out _aspBnScale, out _aspBnShift);
        _aspAttn3W = WhisperOps.EnsureF32(w["pooling.attention.3.weight"]);
        _aspAttn3B = WhisperOps.EnsureF32(w["pooling.attention.3.bias"]);

        _bottleW = WhisperOps.EnsureF32(w["bottleneck.weight"]);
        _bottleB = WhisperOps.EnsureF32(w["bottleneck.bias"]);
        _ldaW = WhisperOps.EnsureF32(lda["weight"]);
        _ldaB = WhisperOps.EnsureF32(lda["bias"]);
    }

    /// <summary>Builds the 128-d speaker embedding from a mono 16 kHz waveform.</summary>
    public Tensor EmbedFromWav(IBackend backend, ReadOnlySpan<float> wav16k)
    {
        (Tensor mel, int t) = BuildMel(wav16k);
        Tensor emb = Embed(backend, mel, t);
        mel.Dispose();
        return emb;
    }

    /// <summary>Computes the logFbank <c>[1, 80, T]</c> (channels-first) with the reference's additive log floor
    /// and per-mel time-mean subtraction.</summary>
    public (Tensor mel, int t) BuildMel(ReadOnlySpan<float> wav16k)
    {
        int nMels = _cfg.NumMels;
        float[,] frames = _mel.Compute(wav16k);          // [nMels, T], already log(mel + 1e-6)
        int t = frames.GetLength(1);
        Tensor mel = new(new TensorShape(1, nMels, t), DType.F32);
        float* mp = (float*)mel.DataPointer;
        // Per-mel time-mean subtraction: out = out - out.mean(dim=time).
        for (int m = 0; m < nMels; m++)
        {
            double mean = 0;
            for (int i = 0; i < t; i++) mean += frames[m, i];
            mean /= t;
            for (int i = 0; i < t; i++) mp[(long)m * t + i] = frames[m, i] - (float)mean;
        }
        return (mel, t);
    }

    /// <summary>Embeds a logFbank mel <c>[1, 80, T]</c> (channels-first) into a <c>[1, 128]</c> speaker vector.</summary>
    public Tensor Embed(IBackend backend, Tensor mel, int t)
    {
        if (_stem is null) throw new InvalidOperationException("ZonosSpeakerEncoder weights not loaded.");
        int nMels = _cfg.NumMels;

        // Treat the [1, 80, T] mel as a single-channel image [1, 1, 80, T].
        Tensor img = mel.Reshape(new TensorShape(1, 1, nMels, t));
        int h = nMels, w = t;

        Tensor x = ApplyConv(backend, img, _stem!, 1, 1, ref h, ref w);
        Relu(backend, x);

        foreach (Stage stage in _stages)
            foreach (SimAmBlock block in stage.Blocks)
            {
                Tensor next = ForwardBlock(backend, x, block, ref h, ref w);
                x.Dispose();
                x = next;
            }

        // Fold the frequency axis into channels: [1, C, H, W] → [1, C*H, W]; ASP pools over W (time).
        int poolIn = _cfg.StageWidths[^1] * h;           // 512 * 10 = 5120
        Tensor pooled = AttentiveStatsPooling(backend, x, poolIn, w);   // [1, 1, 2*poolIn]
        x.Dispose();

        Tensor bottle = WhisperOps.ProjectLinear(backend, pooled, _bottleW!, _bottleB, 1, 1, 2 * poolIn, _cfg.BottleneckDim);
        pooled.Dispose();
        Tensor lda = WhisperOps.ProjectLinear(backend, bottle, _ldaW!, _ldaB, 1, 1, _cfg.BottleneckDim, _cfg.EmbedDim);
        bottle.Dispose();
        return lda.Reshape(new TensorShape(1, _cfg.EmbedDim));
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_stem is not null) foreach (Tensor x in _stem.All()) yield return x;
        foreach (Stage s in _stages)
            foreach (SimAmBlock b in s.Blocks)
                foreach (Tensor x in b.All()) yield return x;
        Tensor?[] tail = [_aspAttn0W, _aspAttn0B, _aspBnScale, _aspBnShift, _aspAttn3W, _aspAttn3B,
            _bottleW, _bottleB, _ldaW, _ldaB];
        foreach (Tensor? x in tail) if (x is not null) yield return x;
    }

    /// <summary>One SimAM BasicBlock: conv1→ReLU → conv2 → SimAM → +shortcut → ReLU (BN is folded into the convs).</summary>
    private Tensor ForwardBlock(IBackend backend, Tensor x, SimAmBlock block, ref int h, ref int w)
    {
        int inH = h, inW = w;
        int outH = h, outW = w;
        Tensor h1 = ApplyConv(backend, x, block.Conv1, block.Stride, 1, ref outH, ref outW);
        Relu(backend, h1);
        int th = outH, tw = outW;
        Tensor h2 = ApplyConv(backend, h1, block.Conv2, 1, 1, ref th, ref tw);
        h1.Dispose();

        SimAm(backend, h2, block.OutChannels, outH, outW);

        Tensor shortcut;
        if (block.Down is not null)
        {
            int sh = inH, sw = inW;
            shortcut = ApplyConv(backend, x, block.Down, block.Stride, 0, ref sh, ref sw);
        }
        else
        {
            shortcut = new Tensor(x.Shape, DType.F32);
            backend.CopyTo(shortcut, x);
        }

        Tensor sum = new(h2.Shape, DType.F32);
        backend.Add(sum, h2, shortcut);
        h2.Dispose();
        shortcut.Dispose();
        Relu(backend, sum);

        h = outH;
        w = outW;
        return sum;
    }

    /// <summary>Runs a folded conv (k3 pad1, or k1 pad0 for a downsample) and returns the output; updates
    /// <paramref name="h"/>/<paramref name="w"/> to the produced spatial size.</summary>
    private static Tensor ApplyConv(IBackend backend, Tensor input, ConvBn conv, int stride, int pad, ref int h, ref int w)
    {
        int k = conv.KernelSize;
        int outH = (h + 2 * pad - k) / stride + 1;
        int outW = (w + 2 * pad - k) / stride + 1;
        Tensor outT = new(new TensorShape(1, conv.OutChannels, outH, outW), DType.F32);
        backend.Conv2D(outT, input, conv.Weight!, conv.Bias, stride, stride, pad, pad);
        h = outH; w = outW;
        return outT;
    }

    /// <summary>SimAM in-place: per-channel energy weighting <c>x·sigmoid(e)</c> where
    /// <c>e = d²/(4·(v+λ)) + 0.5</c>, <c>d = x − mean</c>, <c>v = Σd²/(H·W−1)</c> over each channel plane.</summary>
    private void SimAm(IBackend backend, Tensor x, int c, int h, int w)
    {
        backend.Sync();   // x was just written by a GPU conv; block on it before host reads (avoids async-free race)
        float* p = (float*)x.DataPointer;
        long plane = (long)h * w;
        float n = plane - 1 > 0 ? plane - 1 : 1;
        for (int ch = 0; ch < c; ch++)
        {
            long off = (long)ch * plane;
            float mean = 0f;
            for (long i = 0; i < plane; i++) mean += p[off + i];
            mean /= plane;
            float var = 0f;
            for (long i = 0; i < plane; i++) { float d = p[off + i] - mean; var += d * d; }
            var /= n;
            float denom = 4f * (var + _cfg.SimAmLambda);
            for (long i = 0; i < plane; i++)
            {
                float d = p[off + i] - mean;
                float e = d * d / denom + 0.5f;
                p[off + i] *= 1f / (1f + MathF.Exp(-e));
            }
        }
    }

    /// <summary>Attentive Statistics Pooling: reshape <c>[1, C, H, W]</c> → <c>[1, poolIn=C·H, W]</c>, compute a
    /// per-(pool-channel) time attention <c>softmax(conv3(bn(relu(conv0(x)))))</c>, then return the concatenation
    /// of the attention-weighted mean and std → <c>[1, 1, 2·poolIn]</c>.</summary>
    private Tensor AttentiveStatsPooling(IBackend backend, Tensor x, int poolIn, int w)
    {
        Tensor feat = x.Reshape(new TensorShape(1, poolIn, w));          // [1, 5120, W]

        int attDim = _cfg.AspAttentionDim;
        Tensor a0 = new(new TensorShape(1, attDim, w), DType.F32);
        backend.Conv1d(a0, feat, _aspAttn0W!, _aspAttn0B, 1, 0, 0, 1, 1);
        Relu(backend, a0);
        ApplyBatchNorm1d(a0, attDim, w);
        Tensor scores = new(new TensorShape(1, poolIn, w), DType.F32);
        backend.Conv1d(scores, a0, _aspAttn3W!, _aspAttn3B, 1, 0, 0, 1, 1);
        a0.Dispose();
        backend.Sync();   // scores was just written by the GPU conv; block before the host softmax + stats
        SoftmaxOverTime(scores, poolIn, w);

        float* sp = (float*)scores.DataPointer;   // [poolIn, W] attention weights
        float* xp = (float*)feat.DataPointer;      // [poolIn, W] features
        Tensor outT = new(new TensorShape(1, 1, 2 * poolIn), DType.F32);
        float* op = (float*)outT.DataPointer;
        for (int d = 0; d < poolIn; d++)
        {
            double mu = 0;
            for (int ti = 0; ti < w; ti++) mu += (double)sp[(long)d * w + ti] * xp[(long)d * w + ti];
            double ex2 = 0;
            for (int ti = 0; ti < w; ti++) { double v = xp[(long)d * w + ti]; ex2 += sp[(long)d * w + ti] * v * v; }
            double varv = ex2 - mu * mu;
            if (varv < 1e-5) varv = 1e-5;
            op[d] = (float)mu;
            op[poolIn + d] = (float)Math.Sqrt(varv);
        }
        scores.Dispose();
        return outT;
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

    private void ApplyBatchNorm1d(Tensor x, int c, int t)
    {
        float* xp = (float*)x.DataPointer;
        float* scale = (float*)_aspBnScale!.DataPointer;
        float* shift = (float*)_aspBnShift!.DataPointer;
        for (int ci = 0; ci < c; ci++)
            for (int s = 0; s < t; s++)
            {
                long idx = (long)ci * t + s;
                xp[idx] = xp[idx] * scale[ci] + shift[ci];
            }
    }

    private static void Relu(IBackend backend, Tensor x)
    {
        backend.Sync();   // block on the preceding GPU op before this host read/modify (avoids async-free race)
        float* p = (float*)x.DataPointer;
        long n = x.ElementCount;
        for (long i = 0; i < n; i++) if (p[i] < 0f) p[i] = 0f;
    }

    /// <summary>Folds a Conv2d + BatchNorm2d pair into a single conv with bias: <c>W' = W·γ/√(σ²+ε)</c>,
    /// <c>b' = β − μ·γ/√(σ²+ε)</c>.</summary>
    private ConvBn FoldConvBn(IReadOnlyDictionary<string, Tensor> w, string convKey, string bnKey)
    {
        Tensor rawW = WhisperOps.EnsureF32(w[$"{convKey}.weight"]);   // [Cout, Cin, kH, kW]
        int cout = (int)rawW.Shape[0];
        int cin = (int)rawW.Shape[1];
        int kh = (int)rawW.Shape[2];
        int kw = (int)rawW.Shape[3];
        float* gamma = (float*)WhisperOps.EnsureF32(w[$"{bnKey}.weight"]).DataPointer;
        float* beta = (float*)WhisperOps.EnsureF32(w[$"{bnKey}.bias"]).DataPointer;
        float* mean = (float*)WhisperOps.EnsureF32(w[$"{bnKey}.running_mean"]).DataPointer;
        float* var = (float*)WhisperOps.EnsureF32(w[$"{bnKey}.running_var"]).DataPointer;

        Tensor foldedW = new(rawW.Shape, DType.F32);
        Tensor foldedB = new(new TensorShape(cout), DType.F32);
        float* srcW = (float*)rawW.DataPointer;
        float* dstW = (float*)foldedW.DataPointer;
        float* dstB = (float*)foldedB.DataPointer;
        long perOut = (long)cin * kh * kw;
        for (int co = 0; co < cout; co++)
        {
            float s = gamma[co] / MathF.Sqrt(var[co] + _cfg.BnEps);
            dstB[co] = beta[co] - mean[co] * s;
            long off = co * perOut;
            for (long i = 0; i < perOut; i++) dstW[off + i] = srcW[off + i] * s;
        }
        return new ConvBn { Weight = foldedW, Bias = foldedB, OutChannels = cout, KernelSize = kh };
    }

    /// <summary>Precomputes the folded BatchNorm1d scale/shift: <c>scale = γ/√(σ²+ε)</c>, <c>shift = β − μ·scale</c>.</summary>
    private void FoldBatchNorm1d(IReadOnlyDictionary<string, Tensor> w, string bnKey, out Tensor? scale, out Tensor? shift)
    {
        Tensor gamma = WhisperOps.EnsureF32(w[$"{bnKey}.weight"]);
        int c = (int)gamma.Shape[0];
        float* g = (float*)gamma.DataPointer;
        float* beta = (float*)WhisperOps.EnsureF32(w[$"{bnKey}.bias"]).DataPointer;
        float* mean = (float*)WhisperOps.EnsureF32(w[$"{bnKey}.running_mean"]).DataPointer;
        float* var = (float*)WhisperOps.EnsureF32(w[$"{bnKey}.running_var"]).DataPointer;
        scale = new Tensor(new TensorShape(c), DType.F32);
        shift = new Tensor(new TensorShape(c), DType.F32);
        float* sp = (float*)scale.DataPointer;
        float* hp = (float*)shift.DataPointer;
        for (int i = 0; i < c; i++)
        {
            sp[i] = g[i] / MathF.Sqrt(var[i] + _cfg.BnEps);
            hp[i] = beta[i] - mean[i] * sp[i];
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private sealed class Stage
    {
        public required SimAmBlock[] Blocks { get; init; }
    }

    private sealed class SimAmBlock
    {
        public required ConvBn Conv1 { get; init; }
        public required ConvBn Conv2 { get; init; }
        public ConvBn? Down { get; init; }
        public int Stride { get; init; }
        public int OutChannels { get; init; }

        public IEnumerable<Tensor> All()
        {
            foreach (Tensor t in Conv1.All()) yield return t;
            foreach (Tensor t in Conv2.All()) yield return t;
            if (Down is not null) foreach (Tensor t in Down.All()) yield return t;
        }
    }

    private sealed class ConvBn
    {
        public Tensor? Weight { get; init; }
        public Tensor? Bias { get; init; }
        public int OutChannels { get; init; }
        public int KernelSize { get; init; }

        public IEnumerable<Tensor> All()
        {
            if (Weight is not null) yield return Weight;
            if (Bias is not null) yield return Bias;
        }
    }
}
