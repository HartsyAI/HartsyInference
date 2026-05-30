using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.StyleTts2;

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
    private static readonly float InvSqrt2 = 1f / MathF.Sqrt(2f);
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
        // spectral_norm stores the effective weight under `.weight` in eval-exported dicts; otherwise
        // `.weight_orig`. (Exact sigma folding is checkpoint-gated.)
        if (w.TryGetValue($"{prefix}.weight", out Tensor? wt)) return WhisperOps.EnsureF32(wt);
        return WhisperOps.EnsureF32(w[$"{prefix}.weight_orig"]);
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

    internal static Tensor AvgPool2x(Tensor x)
    {
        int c = (int)x.Shape[1], h = (int)x.Shape[2], w = (int)x.Shape[3];
        int oh = h / 2, ow = w / 2;
        Tensor outT = new(new TensorShape(1, c, oh, ow), DType.F32);
        float* ip = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int cc = 0; cc < c; cc++)
            for (int y = 0; y < oh; y++)
                for (int xx = 0; xx < ow; xx++)
                {
                    long b = (long)cc * h * w;
                    float s = ip[b + (2 * y) * w + 2 * xx] + ip[b + (2 * y) * w + 2 * xx + 1]
                            + ip[b + (2 * y + 1) * w + 2 * xx] + ip[b + (2 * y + 1) * w + 2 * xx + 1];
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

/// <summary>StarGAN-v2 residual block (no normalization): <c>(residual + shortcut)/√2</c> where
/// <c>residual = conv2(actv(avgpool?(conv1(actv(x)))))</c> and <c>shortcut = avgpool?(conv1x1?(x))</c>.
/// Always halves the spatial size (downsample) and may change channels.</summary>
internal sealed unsafe class ResBlk2D
{
    private const float LeakySlope = 0.2f;
    private static readonly float InvSqrt2 = 1f / MathF.Sqrt(2f);
    private readonly int _inCh, _outCh;
    private readonly bool _learnedSc;
    private Tensor? _conv1W, _conv1B, _conv2W, _conv2B, _conv1x1W;

    public ResBlk2D(int inCh, int outCh)
    {
        _inCh = inCh;
        _outCh = outCh;
        _learnedSc = inCh != outCh;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _conv1W = StyleEncoder.ReadConv(w, $"{prefix}.conv1");
        _conv1B = WhisperOps.EnsureF32(w[$"{prefix}.conv1.bias"]);
        _conv2W = StyleEncoder.ReadConv(w, $"{prefix}.conv2");
        _conv2B = WhisperOps.EnsureF32(w[$"{prefix}.conv2.bias"]);
        if (_learnedSc) _conv1x1W = StyleEncoder.ReadConv(w, $"{prefix}.conv1x1");
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        // Residual: actv → conv1(in→out, k3p1) → avgpool → actv → conv2(out→out, k3p1).
        Tensor r = new(x.Shape, DType.F32);
        Buffer.MemoryCopy((void*)x.DataPointer, (void*)r.DataPointer, x.ElementCount * 4, x.ElementCount * 4);
        backend.LeakyRelu(r, r, LeakySlope);
        Tensor c1 = StyleEncoder.Conv2dSame(backend, r, _conv1W!, _conv1B!, _outCh, 3, 1);
        r.Dispose();
        Tensor c1d = StyleEncoder.AvgPool2x(c1);
        c1.Dispose();
        backend.LeakyRelu(c1d, c1d, LeakySlope);
        Tensor residual = StyleEncoder.Conv2dSame(backend, c1d, _conv2W!, _conv2B!, _outCh, 3, 1);
        c1d.Dispose();

        // Shortcut: conv1x1? → avgpool.
        Tensor sc = _learnedSc
            ? StyleEncoder.Conv2dSame(backend, x, _conv1x1W!, bias: null, _outCh, 1, 0)
            : Clone(x);
        Tensor scd = StyleEncoder.AvgPool2x(sc);
        sc.Dispose();

        float* rp = (float*)residual.DataPointer;
        float* sp = (float*)scd.DataPointer;
        long n = residual.ElementCount;
        for (long i = 0; i < n; i++) rp[i] = (rp[i] + sp[i]) * InvSqrt2;
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
        Tensor?[] all = [_conv1W, _conv1B, _conv2W, _conv2B, _conv1x1W];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
