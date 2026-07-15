using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.StyleTts2;

/// <summary>StyleTTS 2 speech encoder — a 2D-Conv ResNet (StarGAN-v2 style) that maps a reference mel
/// <c>[1, 1, 80, T]</c> to a 128-d style vector via a stem Conv → 4 halving ResBlks (64→128→256→512→512)
/// → a 5×5 valid Conv → adaptive average pool → Linear head. Two of these run in parallel with
/// independent weights: <c>style_encoder</c> (acoustic half → decoder) and <c>predictor_encoder</c>
/// (prosodic half → prosody predictor); their outputs concatenate into the 256-d style vector.
///
/// <para><b>Scaffold note:</b> the released weights are <c>spectral_norm</c>-wrapped (<c>weight_orig</c>
/// + <c>weight_u</c>); <see cref="LoadWeights"/> reads the effective <c>.weight</c> and falls back to
/// <c>.weight_orig</c> — spectral-norm sigma folding is checkpoint-gated. ResBlks use no normalization
/// (StarGAN-v2 StyleEncoder convention).</para></summary>
public sealed unsafe class StyleEncoder : IDisposable
{
    private const float LeakySlope = 0.2f;
    private static readonly float _invSqrt2 = 1f / MathF.Sqrt(2f);
    private readonly int _styleDim;
    private int _disposed;

    private Tensor? _stemW, _stemB;
    private readonly ResBlk2D[] _blocks = new ResBlk2D[4];
    private Tensor? _tailW, _tailB;          // Conv2d(512, 512, k5, p0)
    private Tensor? _headW, _headB;          // Linear(512, 128)

    public StyleEncoder(int styleDim = 128)
    {
        _styleDim = styleDim;
        int[] inC = [64, 128, 256, 512];
        int[] outC = [128, 256, 512, 512];
        for (int i = 0; i < 4; i++) _blocks[i] = new ResBlk2D(inC[i], outC[i]);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _stemW = ReadConv(w, $"{p}stem");
        _stemB = WhisperOps.EnsureF32(w[$"{p}stem.bias"]);
        for (int i = 0; i < 4; i++) _blocks[i].LoadWeights(w, $"{p}blocks.{i}");
        _tailW = ReadConv(w, $"{p}tail");
        _tailB = WhisperOps.EnsureF32(w[$"{p}tail.bias"]);
        _headW = WhisperOps.EnsureF32(w[$"{p}unshared.weight"]);
        _headB = WhisperOps.EnsureF32(w[$"{p}unshared.bias"]);
    }

    internal static Tensor ReadConv(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        // Eval-exported dicts store the effective weight under `.weight`; the training checkpoint stores
        // spectral_norm's `weight_orig` + power-iteration vectors `weight_u`/`weight_v`, so fold the spectral
        // norm here: reshape W to [out, in·kh·kw], σ = uᵀ(W·v), W_eff = W_orig / σ (torch spectral_norm).
        if (w.TryGetValue($"{prefix}.weight", out Tensor? wt)) return WhisperOps.EnsureF32(wt);
        Tensor wOrig = WhisperOps.EnsureF32(w[$"{prefix}.weight_orig"]);
        if (!w.TryGetValue($"{prefix}.weight_u", out Tensor? uT) || !w.TryGetValue($"{prefix}.weight_v", out Tensor? vT))
            return wOrig;
        return FoldSpectralNorm(wOrig, WhisperOps.EnsureF32(uT), WhisperOps.EnsureF32(vT));
    }

    /// <summary>Folds torch <c>spectral_norm</c>: σ = uᵀ(W₂ₐ·v) over W reshaped to <c>[out, in·kh·kw]</c>,
    /// returns a new tensor <c>W_orig / σ</c> (same shape as <paramref name="wOrig"/>).</summary>
    internal static Tensor FoldSpectralNorm(Tensor wOrig, Tensor u, Tensor v)
    {
        int outC = (int)wOrig.Shape[0];
        long cols = wOrig.ElementCount / outC;
        float* wp = (float*)wOrig.DataPointer, up = (float*)u.DataPointer, vp = (float*)v.DataPointer;
        // σ = Σ_o u[o] · (Σ_c W[o,c]·v[c])
        double sigma = 0;
        for (int o = 0; o < outC; o++)
        {
            double row = 0;
            long baseOff = (long)o * cols;
            for (long c = 0; c < cols; c++) row += wp[baseOff + c] * vp[c];
            sigma += up[o] * row;
        }
        float inv = sigma != 0 ? (float)(1.0 / sigma) : 1f;
        Tensor outT = new(wOrig.Shape, DType.F32);
        float* op = (float*)outT.DataPointer;
        long n = wOrig.ElementCount;
        for (long i = 0; i < n; i++) op[i] = wp[i] * inv;
        return outT;
    }

    /// <summary>Extracts the 128-d style vector from a reference mel <c>[1, 1, 80, T]</c> (or
    /// <c>[1, 80, T]</c>, reshaped here).</summary>
    public Tensor Forward(IBackend backend, Tensor mel)
    {
        if (_stemW is null) throw new InvalidOperationException("StyleEncoder weights not loaded.");
        Tensor x = mel.Shape.Rank == 3 ? mel.Reshape(new TensorShape(1, 1, mel.Shape[1], mel.Shape[2])) : mel;

        Tensor h = Conv2dSame(backend, x, _stemW!, _stemB!, 64, 3, 1);
        for (int i = 0; i < 4; i++)
        {
            Tensor nh = _blocks[i].Forward(backend, h);
            h.Dispose();
            h = nh;
        }
        backend.LeakyRelu(h, h, LeakySlope);

        // Tail: valid 5×5 conv (no pad) — shrinks H from 5 → 1 and W by 4.
        Tensor tail = Conv2dValid(backend, h, _tailW!, _tailB!, 512, 5);
        h.Dispose();

        // Adaptive avg pool to 1×1 → [1, 512].
        Tensor pooled = GlobalAvgPool(tail, 512);
        tail.Dispose();
        backend.LeakyRelu(pooled, pooled, LeakySlope);

        // Linear head → [1, styleDim].
        Tensor pooled3 = pooled.Reshape(new TensorShape(1, 1, 512));
        Tensor styled = WhisperOps.ProjectLinear(backend, pooled3, _headW!, _headB, 1, 1, 512, _styleDim);
        pooled.Dispose();
        return styled.Reshape(new TensorShape(1, _styleDim));
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] core = [_stemW, _stemB, _tailW, _tailB, _headW, _headB];
        foreach (Tensor? t in core) if (t is not null) yield return t;
        foreach (ResBlk2D b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    // ── 2D conv / pool helpers (channels-first [1, C, H, W]) ──

    internal static Tensor Conv2dSame(IBackend backend, Tensor x, Tensor wgt, Tensor? bias, int outCh, int k, int pad)
    {
        int h = (int)x.Shape[2], wdt = (int)x.Shape[3];
        Tensor outT = new(new TensorShape(1, outCh, h, wdt), DType.F32);
        backend.Conv2D(outT, x, wgt, bias, strideH: 1, strideW: 1, padH: pad, padW: pad);
        return outT;
    }

    internal static Tensor Conv2dValid(IBackend backend, Tensor x, Tensor wgt, Tensor bias, int outCh, int k)
    {
        int h = (int)x.Shape[2] - (k - 1), wdt = (int)x.Shape[3] - (k - 1);
        if (h < 1) h = 1;
        if (wdt < 1) wdt = 1;
        Tensor outT = new(new TensorShape(1, outCh, h, wdt), DType.F32);
        backend.Conv2D(outT, x, wgt, bias, strideH: 1, strideW: 1, padH: 0, padW: 0);
        return outT;
    }

    /// <summary>2× average pool matching StyleTTS2 <c>DownSample('half')</c>: height floors, but an ODD width
    /// replicates its last column before pooling (<c>torch.cat([x, x[…,-1]])</c>), so the width becomes
    /// <c>ceil(w/2)</c> and the boundary output = the last column's value. This keeps the shortcut's size aligned
    /// with the residual's learned stride-2 downsample on odd time frames.</summary>
    internal static Tensor AvgPool2x(Tensor x)
    {
        int c = (int)x.Shape[1], h = (int)x.Shape[2], w = (int)x.Shape[3];
        int oh = h / 2, ow = (w + 1) / 2;   // width: ceil (odd → replicate last column)
        Tensor outT = new(new TensorShape(1, c, oh, ow), DType.F32);
        float* ip = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int cc = 0; cc < c; cc++)
            for (int y = 0; y < oh; y++)
                for (int xx = 0; xx < ow; xx++)
                {
                    long b = (long)cc * h * w;
                    int x0 = 2 * xx, x1 = Math.Min(2 * xx + 1, w - 1);   // clamp = replicate the odd last column
                    float s = ip[b + (2 * y) * w + x0] + ip[b + (2 * y) * w + x1]
                            + ip[b + (2 * y + 1) * w + x0] + ip[b + (2 * y + 1) * w + x1];
                    op[(long)cc * oh * ow + y * ow + xx] = s * 0.25f;
                }
        return outT;
    }

    private static Tensor GlobalAvgPool(Tensor x, int channels)
    {
        int h = (int)x.Shape[2], w = (int)x.Shape[3];
        Tensor outT = new(new TensorShape(1, channels), DType.F32);
        float* ip = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        long hw = (long)h * w;
        for (int cc = 0; cc < channels; cc++)
        {
            double s = 0;
            long b = (long)cc * hw;
            for (long i = 0; i < hw; i++) s += ip[b + i];
            op[cc] = (float)(s / hw);
        }
        return outT;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}

/// <summary>StarGAN-v2 residual block (no normalization), <c>downsample='half'</c>: <c>(residual + shortcut)/√2</c>
/// where <c>residual = conv2(actv(downsample_res(conv1(actv(x)))))</c> — <c>conv1</c> is dim_in→dim_in, the
/// downsample is a LEARNED depthwise stride-2 conv (<c>downsample_res.conv</c>, groups=dim_in), <c>conv2</c> is
/// dim_in→dim_out — and <c>shortcut = avgpool2×(conv1x1?(x))</c> (parameter-free avg-pool). Halves the spatial
/// size and may change channels. All convs are spectral-norm-folded on load.</summary>
internal sealed unsafe class ResBlk2D
{
    private const float LeakySlope = 0.2f;
    private static readonly float _invSqrt2 = 1f / MathF.Sqrt(2f);
    private readonly int _inCh, _outCh;
    private readonly bool _learnedSc;
    private Tensor? _conv1W, _conv1B, _downW, _downB, _conv2W, _conv2B, _conv1x1W;

    public ResBlk2D(int inCh, int outCh)
    {
        _inCh = inCh;
        _outCh = outCh;
        _learnedSc = inCh != outCh;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _conv1W = StyleEncoder.ReadConv(w, $"{prefix}.conv1");                       // [in, in, 3, 3]
        _conv1B = WhisperOps.EnsureF32(w[$"{prefix}.conv1.bias"]);
        _downW = StyleEncoder.ReadConv(w, $"{prefix}.downsample_res.conv");          // [in, 1, 3, 3] depthwise
        _downB = WhisperOps.EnsureF32(w[$"{prefix}.downsample_res.conv.bias"]);
        _conv2W = StyleEncoder.ReadConv(w, $"{prefix}.conv2");                       // [out, in, 3, 3]
        _conv2B = WhisperOps.EnsureF32(w[$"{prefix}.conv2.bias"]);
        if (_learnedSc) _conv1x1W = StyleEncoder.ReadConv(w, $"{prefix}.conv1x1");   // [out, in, 1, 1], no bias
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        int iH = (int)x.Shape[2], iW = (int)x.Shape[3];
        int oH = (iH + 2 - 3) / 2 + 1, oW = (iW + 2 - 3) / 2 + 1;   // stride-2, k3, pad1

        // Residual: actv → conv1(in→in, k3p1) → learned depthwise downsample(stride2) → actv → conv2(in→out, k3p1).
        Tensor r = Clone(x);
        backend.LeakyRelu(r, r, LeakySlope);
        Tensor c1 = StyleEncoder.Conv2dSame(backend, r, _conv1W!, _conv1B!, _inCh, 3, 1);
        r.Dispose();
        Tensor c1d = new(new TensorShape(1, _inCh, oH, oW), DType.F32);
        backend.Conv2dDepthwise(c1d, c1, _downW!, _downB!, strideH: 2, strideW: 2, padH: 1, padW: 1);
        c1.Dispose();
        backend.LeakyRelu(c1d, c1d, LeakySlope);
        Tensor residual = StyleEncoder.Conv2dSame(backend, c1d, _conv2W!, _conv2B!, _outCh, 3, 1);
        c1d.Dispose();

        // Shortcut: conv1x1? → avgpool2×.
        Tensor sc = _learnedSc
            ? StyleEncoder.Conv2dSame(backend, x, _conv1x1W!, bias: null, _outCh, 1, 0)
            : Clone(x);
        Tensor scd = StyleEncoder.AvgPool2x(sc);
        sc.Dispose();

        float* rp = (float*)residual.DataPointer;
        float* sp = (float*)scd.DataPointer;
        long n = residual.ElementCount;
        for (long i = 0; i < n; i++) rp[i] = (rp[i] + sp[i]) * _invSqrt2;
        scd.Dispose();
        return residual;
    }

    private static Tensor Clone(Tensor x)
    {
        Tensor c = new(x.Shape, DType.F32);
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)c.DataPointer, x.ElementCount * 4, x.ElementCount * 4);
        return c;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_conv1W, _conv1B, _downW, _downB, _conv2W, _conv2B, _conv1x1W];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
