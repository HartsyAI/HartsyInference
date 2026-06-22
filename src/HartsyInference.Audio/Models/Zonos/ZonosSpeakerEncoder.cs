using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Zonos;

/// <summary>Zonos speaker encoder (the separate-download "spk_clf" model): an 80-mel logFbank front-end →
/// ResNet293 trunk (4 stages of bottleneck residual blocks with SimAM attention) → Attentive Statistics
/// Pooling (ASP) over the time axis → a linear bottleneck to 256-d → an LDA linear projection to the final
/// 128-d speaker embedding consumed by the Zonos prefix conditioner.
///
/// <para>Reuses the shared <see cref="IBackend"/> Conv2D / GroupNorm / ScaledDotProductAttention-free ops and
/// <see cref="WhisperOps"/> linear/transpose helpers. The mel is supplied channels-first <c>[1, 80, T]</c>;
/// internally it is treated as a single-channel 2-D image <c>[1, 1, 80, T]</c> for the conv trunk (the ResNet
/// convolves over the (freq, time) plane), then ASP collapses time to a fixed-size statistics vector.</para>
///
/// <para>Structural scaffold (validation-pending): the exact bottleneck width schedule and the SimAM lambda are
/// captured from <c>docs/Research/ZONOS_ARCHITECTURE.md</c>; the precise checkpoint key spelling and the
/// channel-multiplier ("base_width") are reconcile items against the downloaded <c>speaker_embedding</c>
/// model. The forward math (SimAM, ASP, LDA) is exact.</para></summary>
public sealed unsafe class ZonosSpeakerEncoder : IDisposable
{
    private readonly ZonosSpeakerConfig _cfg;
    private readonly Stage[] _stages;

    private Tensor? _stemW, _stemB;            // Conv2d(1 → baseWidth, k3, p1)
    private Tensor? _stemNormW, _stemNormB;    // GroupNorm
    private Tensor? _aspAttnW, _aspAttnB;      // ASP attention Linear(poolIn → poolIn) (tanh) + score Linear(poolIn → poolIn)
    private Tensor? _aspScoreW, _aspScoreB;
    private Tensor? _bottleW, _bottleB;        // Linear(2*poolIn → 256)
    private Tensor? _bottleNormW, _bottleNormB; // BatchNorm/LayerNorm over 256 (affine)
    private Tensor? _ldaW, _ldaB;              // Linear(256 → 128)
    private int _disposed;

    public ZonosSpeakerConfig Config => _cfg;

    public ZonosSpeakerEncoder(ZonosSpeakerConfig cfg)
    {
        _cfg = cfg;
        _stages = new Stage[cfg.StageBlocks.Count];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "speaker_encoder")
    {
        _stemW = ConvW(w[$"{prefix}.stem.conv.weight"]);
        _stemB = TryGet(w, $"{prefix}.stem.conv.bias");
        _stemNormW = WhisperOps.EnsureF32(w[$"{prefix}.stem.norm.weight"]);
        _stemNormB = WhisperOps.EnsureF32(w[$"{prefix}.stem.norm.bias"]);

        int inC = _cfg.BaseWidth;
        for (int s = 0; s < _stages.Length; s++)
        {
            int outC = _cfg.StageWidths[s];
            int stride = s == 0 ? 1 : 2;
            ResBlock[] blocks = new ResBlock[_cfg.StageBlocks[s]];
            for (int b = 0; b < blocks.Length; b++)
            {
                string bp = $"{prefix}.stages.{s}.blocks.{b}";
                int blockStride = b == 0 ? stride : 1;
                int blockIn = b == 0 ? inC : outC;
                blocks[b] = new ResBlock
                {
                    Conv1W = ConvW(w[$"{bp}.conv1.weight"]),
                    Conv1B = TryGet(w, $"{bp}.conv1.bias"),
                    Norm1W = WhisperOps.EnsureF32(w[$"{bp}.norm1.weight"]),
                    Norm1B = WhisperOps.EnsureF32(w[$"{bp}.norm1.bias"]),
                    Conv2W = ConvW(w[$"{bp}.conv2.weight"]),
                    Conv2B = TryGet(w, $"{bp}.conv2.bias"),
                    Norm2W = WhisperOps.EnsureF32(w[$"{bp}.norm2.weight"]),
                    Norm2B = WhisperOps.EnsureF32(w[$"{bp}.norm2.bias"]),
                    DownW = (blockStride != 1 || blockIn != outC) ? ConvW(w[$"{bp}.downsample.weight"]) : null,
                    DownB = (blockStride != 1 || blockIn != outC) ? TryGet(w, $"{bp}.downsample.bias") : null,
                    Stride = blockStride,
                    OutChannels = outC,
                };
            }
            _stages[s] = new Stage { Blocks = blocks };
            inC = outC;
        }

        _aspAttnW = WhisperOps.EnsureF32(w[$"{prefix}.asp.attention.weight"]);
        _aspAttnB = TryGet(w, $"{prefix}.asp.attention.bias");
        _aspScoreW = WhisperOps.EnsureF32(w[$"{prefix}.asp.score.weight"]);
        _aspScoreB = TryGet(w, $"{prefix}.asp.score.bias");
        _bottleW = WhisperOps.EnsureF32(w[$"{prefix}.bottleneck.weight"]);
        _bottleB = TryGet(w, $"{prefix}.bottleneck.bias");
        _bottleNormW = TryGet(w, $"{prefix}.bottleneck_norm.weight");
        _bottleNormB = TryGet(w, $"{prefix}.bottleneck_norm.bias");
        _ldaW = WhisperOps.EnsureF32(w[$"{prefix}.lda.weight"]);
        _ldaB = TryGet(w, $"{prefix}.lda.bias");
    }

    /// <summary>Embeds a logFbank mel <c>[1, 80, T]</c> (channels-first) into a <c>[1, 128]</c> speaker
    /// vector. <paramref name="t"/> is the valid frame count.</summary>
    public Tensor Embed(IBackend backend, Tensor mel, int t)
    {
        if (_stemW is null) throw new InvalidOperationException("ZonosSpeakerEncoder weights not loaded.");
        int nMels = _cfg.NumMels;

        // Treat the [1, 80, T] mel as a 1-channel image [1, 1, 80, T].
        Tensor img = mel.Reshape(new TensorShape(1, 1, nMels, t));
        int h = nMels, wDim = t;

        // Stem conv (k3, s1, p1) → GroupNorm → ReLU.
        Tensor x = new(new TensorShape(1, _cfg.BaseWidth, h, wDim), DType.F32);
        backend.Conv2D(x, img, _stemW!, _stemB, 1, 1, 1, 1);
        ApplyNormRelu(backend, x, _stemNormW!, _stemNormB!);

        int curC = _cfg.BaseWidth;
        foreach (Stage stage in _stages)
        {
            foreach (ResBlock block in stage.Blocks)
            {
                Tensor next = ForwardBlock(backend, x, block, ref h, ref wDim);
                x.Dispose();
                x = next;
                curC = block.OutChannels;
            }
        }

        // Collapse the frequency axis into the channel axis so ASP pools over time only:
        // [1, C, H, W] → [1, C*H, W].
        int poolIn = curC * h;
        Tensor pooled = AttentiveStatsPooling(backend, x, poolIn, wDim);
        x.Dispose();

        // Bottleneck Linear(2*poolIn → 256) + optional affine norm.
        Tensor bottle = WhisperOps.ProjectLinear(backend, pooled, _bottleW!, _bottleB, 1, 1, 2 * poolIn, _cfg.BottleneckDim);
        pooled.Dispose();
        if (_bottleNormW is not null && _bottleNormB is not null)
        {
            Tensor normed = new(bottle.Shape, DType.F32);
            backend.LayerNorm(normed, bottle, _bottleNormW, _bottleNormB, 1e-5f);
            bottle.Dispose();
            bottle = normed;
        }

        // LDA Linear(256 → 128).
        Tensor lda = WhisperOps.ProjectLinear(backend, bottle, _ldaW!, _ldaB, 1, 1, _cfg.BottleneckDim, _cfg.EmbedDim);
        bottle.Dispose();
        return lda.Reshape(new TensorShape(1, _cfg.EmbedDim));
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top = [_stemW, _stemB, _stemNormW, _stemNormB, _aspAttnW, _aspAttnB, _aspScoreW, _aspScoreB,
            _bottleW, _bottleB, _bottleNormW, _bottleNormB, _ldaW, _ldaB];
        foreach (Tensor? x in top) if (x is not null) yield return x;
        foreach (Stage s in _stages)
            foreach (ResBlock b in s.Blocks)
                foreach (Tensor x in b.All()) yield return x;
    }

    /// <summary>One bottleneck residual block: Conv2d → norm → ReLU → Conv2d → norm → SimAM → +shortcut → ReLU.
    /// The first conv carries the stride; the shortcut downsamples to match when shapes differ.</summary>
    private Tensor ForwardBlock(IBackend backend, Tensor x, ResBlock block, ref int h, ref int w)
    {
        int outH = (h + 2 * 1 - 3) / block.Stride + 1;
        int outW = (w + 2 * 1 - 3) / block.Stride + 1;

        Tensor h1 = new(new TensorShape(1, block.OutChannels, outH, outW), DType.F32);
        backend.Conv2D(h1, x, block.Conv1W!, block.Conv1B, block.Stride, block.Stride, 1, 1);
        ApplyNormRelu(backend, h1, block.Norm1W!, block.Norm1B!);

        Tensor h2 = new(new TensorShape(1, block.OutChannels, outH, outW), DType.F32);
        backend.Conv2D(h2, h1, block.Conv2W!, block.Conv2B, 1, 1, 1, 1);
        h1.Dispose();
        Tensor n2 = new(h2.Shape, DType.F32);
        backend.GroupNorm(n2, h2, block.Norm2W!, block.Norm2B!, _cfg.NormGroups, 1e-5f);
        h2.Dispose();

        // SimAM attention (parameter-free spatial gating).
        SimAm(n2, block.OutChannels, outH, outW);

        // Shortcut.
        Tensor shortcut;
        if (block.DownW is not null)
        {
            shortcut = new Tensor(new TensorShape(1, block.OutChannels, outH, outW), DType.F32);
            backend.Conv2D(shortcut, x, block.DownW!, block.DownB, block.Stride, block.Stride, 0, 0);
        }
        else
        {
            shortcut = new Tensor(x.Shape, DType.F32);
            backend.CopyTo(shortcut, x);
        }

        Tensor sum = new(n2.Shape, DType.F32);
        backend.Add(sum, n2, shortcut);
        n2.Dispose();
        shortcut.Dispose();
        Relu(backend, sum);

        h = outH;
        w = outW;
        return sum;
    }

    /// <summary>SimAM in-place: per-channel energy weighting <c>x * sigmoid(e)</c> where
    /// <c>e = (x - mean)^2 / (4*(var + lambda)) + 0.5</c> over the (H,W) plane of each channel.</summary>
    private void SimAm(Tensor x, int c, int h, int w)
    {
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
                float gate = 1f / (1f + MathF.Exp(-e));
                p[off + i] *= gate;
            }
        }
    }

    /// <summary>Attentive Statistics Pooling: reshapes <c>[1, C, H, W]</c> to <c>[1, C*H, W]</c>, computes a
    /// per-frame attention weight via <c>score(tanh(attn(x)))</c> softmaxed over W, then returns the
    /// concatenation of the attention-weighted mean and standard deviation → <c>[1, 2*poolIn]</c>.</summary>
    private Tensor AttentiveStatsPooling(IBackend backend, Tensor x, int poolIn, int w)
    {
        // x is [1, C, H, W]; reinterpret contiguous data as [1, poolIn, W] (C*H folded into channels).
        Tensor xCf = x.Reshape(new TensorShape(1, poolIn, w));
        // To channels-last [1, W, poolIn] for the attention Linear.
        Tensor xCl = new(new TensorShape(1, w, poolIn), DType.F32);
        backend.Transpose2D(xCl, xCf, poolIn, w);

        Tensor attn = WhisperOps.ProjectLinear(backend, xCl, _aspAttnW!, _aspAttnB, 1, w, poolIn, poolIn);
        Tensor attnAct = new(attn.Shape, DType.F32);
        backend.Tanh(attnAct, attn);
        attn.Dispose();
        Tensor score = WhisperOps.ProjectLinear(backend, attnAct, _aspScoreW!, _aspScoreB, 1, w, poolIn, poolIn);
        attnAct.Dispose();

        // Softmax over the time axis (W) per pool-channel.
        float* sp = (float*)score.DataPointer;   // [W, poolIn]
        for (int d = 0; d < poolIn; d++)
        {
            float maxV = float.NegativeInfinity;
            for (int ti = 0; ti < w; ti++) { float v = sp[(long)ti * poolIn + d]; if (v > maxV) maxV = v; }
            float sum = 0f;
            for (int ti = 0; ti < w; ti++) { float e = MathF.Exp(sp[(long)ti * poolIn + d] - maxV); sp[(long)ti * poolIn + d] = e; sum += e; }
            float inv = sum > 0 ? 1f / sum : 0f;
            for (int ti = 0; ti < w; ti++) sp[(long)ti * poolIn + d] *= inv;
        }

        // Weighted mean and std → [1, 1, 2*poolIn]. Rank-3 (batch, seqLen=1, feature) is required so the
        // downstream WhisperOps.ProjectLinear / BatchedMatMul reads seqLen from Shape[1] and the feature
        // dim from Shape[2]; a rank-2 [1, 2*poolIn] would make BatchedMatMul treat 2*poolIn as the row
        // count M and write an M×outDim result into the (much smaller) output buffer (heap corruption).
        Tensor outT = new(new TensorShape(1, 1, 2 * poolIn), DType.F32);
        float* op = (float*)outT.DataPointer;
        float* xp = (float*)xCl.DataPointer;       // [W, poolIn]
        for (int d = 0; d < poolIn; d++)
        {
            float mean = 0f;
            for (int ti = 0; ti < w; ti++) mean += sp[(long)ti * poolIn + d] * xp[(long)ti * poolIn + d];
            float ex2 = 0f;
            for (int ti = 0; ti < w; ti++) { float v = xp[(long)ti * poolIn + d]; ex2 += sp[(long)ti * poolIn + d] * v * v; }
            float varv = ex2 - mean * mean;
            if (varv < 1e-9f) varv = 1e-9f;
            op[d] = mean;
            op[poolIn + d] = MathF.Sqrt(varv);
        }
        score.Dispose();
        xCl.Dispose();
        return outT;
    }

    private void ApplyNormRelu(IBackend backend, Tensor x, Tensor nw, Tensor nb)
    {
        Tensor n = new(x.Shape, DType.F32);
        backend.GroupNorm(n, x, nw, nb, _cfg.NormGroups, 1e-5f);
        Relu(backend, n);
        backend.CopyTo(x, n);
        n.Dispose();
    }

    private static void Relu(IBackend backend, Tensor x)
    {
        backend.LeakyRelu(x, x, 0f);
    }

    private static Tensor ConvW(Tensor raw) => WhisperOps.EnsureF32(raw);

    private static Tensor? TryGet(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private sealed class Stage
    {
        public required ResBlock[] Blocks { get; init; }
    }

    private sealed class ResBlock
    {
        public Tensor? Conv1W, Conv1B, Norm1W, Norm1B, Conv2W, Conv2B, Norm2W, Norm2B, DownW, DownB;
        public int Stride, OutChannels;
        public IEnumerable<Tensor> All()
        {
            Tensor?[] a = [Conv1W, Conv1B, Norm1W, Norm1B, Conv2W, Conv2B, Norm2W, Norm2B, DownW, DownB];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }
}
