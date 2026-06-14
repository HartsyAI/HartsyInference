using SharpInference.Audio.Layers;
using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Kokoro;

/// <summary>StyleTTS 2's <c>AdainResBlk1d</c> — a style-conditioned residual block used
/// throughout Kokoro's prosody predictor (F0 + Energy chains) and the iSTFTNet decoder.
/// Two AdaIN1d + LeakyReLU + Conv1D substeps, with an optional 2× depthwise transposed-
/// conv upsampler between the first AdaIN and the first conv, and an optional
/// learned 1×1 shortcut on the skip path. Output is averaged via <c>1/√2</c> so the
/// variance stays unit at random initialization.
///
/// <code>
///   residual(x, s):
///     x = AdaIN1d(x, s, dim=dim_in)
///     x = LeakyReLU(0.2)(x)
///     if upsample: x = pool(x)               # depthwise convT(stride=2, k=3) → 2× T
///     x = conv1(x)                            # dim_in → dim_out, k=3
///     x = AdaIN1d(x, s, dim=dim_out)
///     x = LeakyReLU(0.2)(x)
///     x = conv2(x)                            # dim_out → dim_out, k=3
///
///   shortcut(x):
///     if upsample: x = NearestUpsample(x, 2)
///     if learned_sc: x = conv1x1(x)           # dim_in → dim_out, k=1
///
///   forward(x, s):
///     return (residual + shortcut) / √2
/// </code>
///
/// <para>The block is "shape-aware" — it inspects the loaded weights' channel counts
/// (and presence of <c>pool</c> / <c>conv1x1</c> keys) to figure out whether this
/// instance is upsampling and/or has a learned shortcut.</para></summary>
internal sealed unsafe class KokoroAdainResBlk1d
{
    private readonly int _dimIn;
    private readonly int _dimOut;
    private readonly bool _upsample;
    private readonly bool _learnedSc;
    private const int Kernel = 3;
    private const float LeakySlope = 0.2f;
    private static readonly float _invSqrt2 = 1f / MathF.Sqrt(2f);

    // AdaIN1d projections: fc.weight [2*dim, style_dim] + fc.bias [2*dim].
    private Tensor? _norm1FcW, _norm1FcB;
    private Tensor? _norm2FcW, _norm2FcB;

    // Conv1Ds (weights composed from weight_g + weight_v).
    private Tensor? _conv1W, _conv1B;     // dim_in → dim_out, k=3
    private Tensor? _conv2W, _conv2B;     // dim_out → dim_out, k=3
    private Tensor? _conv1x1W;             // dim_in → dim_out, k=1 (only if learnedSc)

    // Pool: depthwise transposed conv with weight [dim_in, 1, 3] + bias [dim_in] (only if upsample).
    private Tensor? _poolW, _poolB;

    public KokoroAdainResBlk1d(int dimIn, int dimOut, bool upsample)
    {
        _dimIn = dimIn;
        _dimOut = dimOut;
        _upsample = upsample;
        _learnedSc = dimIn != dimOut;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _norm1FcW = WhisperOps.EnsureF32(w[$"{prefix}.norm1.fc.weight"]);
        _norm1FcB = WhisperOps.EnsureF32(w[$"{prefix}.norm1.fc.bias"]);
        _norm2FcW = WhisperOps.EnsureF32(w[$"{prefix}.norm2.fc.weight"]);
        _norm2FcB = WhisperOps.EnsureF32(w[$"{prefix}.norm2.fc.bias"]);

        _conv1W = WeightNorm.Compose(w, $"{prefix}.conv1");
        _conv1B = WhisperOps.EnsureF32(w[$"{prefix}.conv1.bias"]);
        _conv2W = WeightNorm.Compose(w, $"{prefix}.conv2");
        _conv2B = WhisperOps.EnsureF32(w[$"{prefix}.conv2.bias"]);

        if (_learnedSc) _conv1x1W = WeightNorm.Compose(w, $"{prefix}.conv1x1");
        if (_upsample)
        {
            _poolW = WeightNorm.Compose(w, $"{prefix}.pool");
            _poolB = WhisperOps.EnsureF32(w[$"{prefix}.pool.bias"]);
        }
    }

    /// <summary>Forward pass over a channels-first <c>[1, dim_in, T]</c> input. Returns
    /// a fresh <c>[1, dim_out, T_out]</c> tensor where <c>T_out = 2*T</c> if
    /// <c>upsample</c>, else <c>T</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor x, Tensor stylePred)
    {
        int batch = (int)x.Shape[0];
        int t = (int)x.Shape[2];
        int tOut = _upsample ? 2 * t : t;
        int pad = Kernel / 2;

        // ── Residual branch ─────────────────────────────────────────────────
        // 1. AdaIN1d on dim_in channels.
        (Tensor gamma1, Tensor beta1) = KokoroOps.StyleToGammaBeta(backend, _norm1FcW!, _norm1FcB!, stylePred, _dimIn);
        Tensor r1 = new(x.Shape, DType.F32);
        backend.AdaInstanceNorm1d(r1, x, gamma1, beta1, eps: 1e-5f);
        gamma1.Dispose(); beta1.Dispose();

        // 2. LeakyReLU.
        backend.LeakyRelu(r1, r1, LeakySlope);

        // 3. Pool (upsample by 2× via depthwise convT, when present).
        Tensor afterPool;
        if (_upsample)
        {
            afterPool = KokoroOps.DepthwiseConvTranspose1d(r1, _poolW!, _poolB, stride: 2, padLeft: 1, padRight: 1, outputPadding: 1);
            r1.Dispose();
        }
        else
        {
            afterPool = r1;
        }

        // 4. Conv1: dim_in → dim_out, k=3, pad=1.
        Tensor c1 = new(new TensorShape(batch, _dimOut, tOut), DType.F32);
        backend.Conv1d(c1, afterPool, _conv1W!, _conv1B, stride: 1, padLeft: pad, padRight: pad, dilation: 1, groups: 1);
        afterPool.Dispose();

        // 5. AdaIN1d on dim_out channels.
        (Tensor gamma2, Tensor beta2) = KokoroOps.StyleToGammaBeta(backend, _norm2FcW!, _norm2FcB!, stylePred, _dimOut);
        Tensor r2 = new(c1.Shape, DType.F32);
        backend.AdaInstanceNorm1d(r2, c1, gamma2, beta2, eps: 1e-5f);
        gamma2.Dispose(); beta2.Dispose();
        c1.Dispose();

        // 6. LeakyReLU.
        backend.LeakyRelu(r2, r2, LeakySlope);

        // 7. Conv2: dim_out → dim_out, k=3, pad=1.
        Tensor residual = new(r2.Shape, DType.F32);
        backend.Conv1d(residual, r2, _conv2W!, _conv2B, stride: 1, padLeft: pad, padRight: pad, dilation: 1, groups: 1);
        r2.Dispose();

        // ── Shortcut branch ─────────────────────────────────────────────────
        Tensor shortcut;
        if (_upsample)
        {
            Tensor up = KokoroOps.NearestUpsample1d(x, 2);
            if (_learnedSc)
            {
                shortcut = new(new TensorShape(batch, _dimOut, tOut), DType.F32);
                backend.Conv1d(shortcut, up, _conv1x1W!, bias: null, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
                up.Dispose();
            }
            else
            {
                shortcut = up;
            }
        }
        else if (_learnedSc)
        {
            shortcut = new(new TensorShape(batch, _dimOut, t), DType.F32);
            backend.Conv1d(shortcut, x, _conv1x1W!, bias: null, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        }
        else
        {
            // Bare identity. We don't own x — make a copy so the caller's lifecycle is
            // unaffected and the caller can dispose its own input freely.
            shortcut = new(x.Shape, DType.F32);
            Buffer.MemoryCopy((void*)x.DataPointer, (void*)shortcut.DataPointer, x.ElementCount * 4, x.ElementCount * 4);
        }

        // ── Combine: (residual + shortcut) / √2 ─────────────────────────────
        Tensor output = new(residual.Shape, DType.F32);
        float* op = (float*)output.DataPointer;
        float* rp = (float*)residual.DataPointer;
        float* sp = (float*)shortcut.DataPointer;
        long n = residual.ElementCount;
        for (long i = 0; i < n; i++) op[i] = (rp[i] + sp[i]) * _invSqrt2;
        residual.Dispose();
        shortcut.Dispose();
        return output;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_norm1FcW, _norm1FcB, _norm2FcW, _norm2FcB,
                         _conv1W, _conv1B, _conv2W, _conv2B, _conv1x1W, _poolW, _poolB];
        foreach (Tensor? x in all) if (x is not null) yield return x;
    }
}
