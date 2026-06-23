using HartsyInference.Audio.Layers;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>S3 speech tokenizer (CosyVoice2 <c>speech_tokenizer_v2_25hz</c>) — converts a reference clip's
/// 128-bin mel into the discrete 25 Hz FSQ token stream the LM/flow consume. A Whisper-style encoder (two
/// stride-2 Conv1d layers, 4× subsample 100→25 Hz, then 6 pre-norm Transformer blocks with rotate-half RoPE
/// and an FSMN memory branch) feeds an 8-channel Finite Scalar Quantizer that packs to <c>3^8 = 6561</c> codes.
/// Inference / encode only.
///
/// <para>Faithful to the upstream <c>s3tokenizer</c> <c>S3TokenizerV2</c> (which is token-identical to
/// CosyVoice2's bundled <c>speech_tokenizer_v2.onnx</c>). Weights load from that model's clean state-dict names
/// (<c>encoder.*</c> / <c>quantizer._codebook.project_down.*</c>); the anonymized ONNX export can be converted
/// to a clean checkpoint with the <c>s3tokenizer</c> package. Convs / attention / norms route through
/// <see cref="IBackend"/>; RoPE, the FSMN residual, and the FSQ packing are CPU sweeps (one-shot reference pass).</para></summary>
public sealed unsafe class S3Tokenizer : IDisposable
{
    private const int NState = 1280;
    private const int NHead = 20;
    private const int HeadDim = 64;           // 1280 / 20
    private const int NLayer = 6;
    private const int FsmnKernel = 31;
    private const int FsqDim = 8;
    private const int FsqLevels = 3;
    private const float FsqScale = 0.9990000128746033f;
    private const float RopeTheta = 10_000f;
    private int _disposed;

    private Tensor? _conv1W, _conv1B, _conv2W, _conv2B;
    private readonly S3Block[] _blocks = new S3Block[NLayer];
    private Tensor? _projDownW, _projDownB;

    public int VocabSize { get; }

    public S3Tokenizer()
    {
        int v = 1;
        for (int i = 0; i < FsqDim; i++) v *= FsqLevels;     // 6561
        VocabSize = v;
        for (int i = 0; i < NLayer; i++) _blocks[i] = new S3Block();
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _conv1W = WhisperOps.EnsureF32(w[$"{p}encoder.conv1.weight"]);
        _conv1B = WhisperOps.EnsureF32(w[$"{p}encoder.conv1.bias"]);
        _conv2W = WhisperOps.EnsureF32(w[$"{p}encoder.conv2.weight"]);
        _conv2B = WhisperOps.EnsureF32(w[$"{p}encoder.conv2.bias"]);
        for (int i = 0; i < NLayer; i++) _blocks[i].LoadWeights(w, $"{p}encoder.blocks.{i}");
        _projDownW = WhisperOps.EnsureF32(w[$"{p}quantizer._codebook.project_down.weight"]);
        _projDownB = WhisperOps.EnsureF32(w[$"{p}quantizer._codebook.project_down.bias"]);
    }

    /// <summary>Encodes a 100 Hz mel <c>[1, 128, T]</c> into 25 Hz speech-token IDs (one per 4 mel frames).</summary>
    public int[] Forward(IBackend backend, Tensor mel)
    {
        if (_projDownW is null) throw new InvalidOperationException("S3Tokenizer weights not loaded.");

        // Conv subsample: two stride-2 Conv1d (k3, pad1) each followed by exact-erf GELU.
        Tensor c1 = Conv(backend, mel, _conv1W!, _conv1B!);
        Activations.ErfGelu(c1);
        Tensor c2 = Conv(backend, c1, _conv2W!, _conv2B!); c1.Dispose();
        Activations.ErfGelu(c2);

        int t = (int)c2.Shape[2];
        Tensor seq = new(new TensorShape(1, t, NState), DType.F32);     // [1, T', 1280] channels-last
        backend.Transpose2D(seq, c2, NState, t);
        c2.Dispose();

        (float[] cos, float[] sin) = RopeTables(t);
        for (int i = 0; i < NLayer; i++)
        {
            Tensor next = _blocks[i].Forward(backend, seq, t, cos, sin);
            seq.Dispose();
            seq = next;
        }

        // FSQ: project_down → tanh → scale → round → +1 → base-3 pack.
        Tensor z = WhisperOps.ProjectLinear(backend, seq, _projDownW!, _projDownB, 1, t, NState, FsqDim);
        seq.Dispose();
        int[] tokens = PackFsqTokens(z, FsqDim, FsqLevels);
        z.Dispose();
        return tokens;
    }

    private static Tensor Conv(IBackend backend, Tensor x, Tensor w, Tensor b)
    {
        int outC = (int)w.Shape[0], k = (int)w.Shape[2], inLen = (int)x.Shape[2];
        int pad = (k - 1) / 2;
        int outLen = (inLen + 2 * pad - (k - 1) - 1) / 2 + 1;     // stride 2
        Tensor o = new(new TensorShape(1, outC, outLen), DType.F32);
        backend.Conv1d(o, x, w, b, 2, pad, pad, 1, 1);
        return o;
    }

    /// <summary>FSQ pack (<c>S3TokenizerV2</c>): per frame <c>token = Σ_j (round(0.999·tanh(z_j)) + (L-1)/2)·L^j</c>.
    /// <paramref name="z"/> is the projected latent <c>[1, T, dim]</c>; returns one base-<paramref name="levels"/>
    /// code per frame. The 0.999 scale and round are verbatim from the level-3 FSQ codebook.</summary>
    public static int[] PackFsqTokens(Tensor z, int dim, int levels)
    {
        int t = (int)z.Shape[1];
        if ((int)z.Shape[2] != dim) throw new ArgumentException($"FSQ latent must have {dim} channels, got {z.Shape[2]}.");
        int halfShift = (levels - 1) / 2;
        float* zp = (float*)z.DataPointer;
        int[] tokens = new int[t];
        for (int frame = 0; frame < t; frame++)
        {
            int token = 0, pow = 1;
            for (int j = 0; j < dim; j++)
            {
                int h = (int)MathF.Round(MathF.Tanh(zp[(long)frame * dim + j]) * FsqScale) + halfShift;
                if (h < 0) h = 0; else if (h > levels - 1) h = levels - 1;
                token += h * pow;
                pow *= levels;
            }
            tokens[frame] = token;
        }
        return tokens;
    }

    /// <summary>Rotate-half RoPE tables: <c>cos/sin[t*32 + i] = cos/sin(t · 10000^(-2i/64))</c>, i in 0..31.</summary>
    private static (float[] Cos, float[] Sin) RopeTables(int t)
    {
        int half = HeadDim / 2;     // 32
        float[] cos = new float[t * half], sin = new float[t * half];
        for (int p = 0; p < t; p++)
            for (int i = 0; i < half; i++)
            {
                double freq = 1.0 / Math.Pow(RopeTheta, (2.0 * i) / HeadDim);
                double ang = p * freq;
                cos[p * half + i] = (float)Math.Cos(ang);
                sin[p * half + i] = (float)Math.Sin(ang);
            }
        return (cos, sin);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top = [_conv1W, _conv1B, _conv2W, _conv2B, _projDownW, _projDownB];
        foreach (Tensor? t in top) if (t is not null) yield return t;
        foreach (S3Block b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}

/// <summary>One <c>S3TokenizerV2</c> encoder block: pre-norm FSMN multi-head self-attention (<c>out(wv) +
/// fsmn(value)</c>, rotate-half RoPE on q/k, attention scale <c>1/√headDim</c>) then a pre-norm GELU MLP,
/// both residual. Channels-last <c>[1, T, 1280]</c>.</summary>
internal sealed unsafe class S3Block
{
    private const int NState = 1280, NHead = 20, HeadDim = 64, FsmnKernel = 31;

    private Tensor? _attnLnW, _attnLnB, _qW, _qB, _kW, _vW, _vB, _oW, _oB, _fsmnW;
    private Tensor? _mlpLnW, _mlpLnB, _mlp0W, _mlp0B, _mlp2W, _mlp2B;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _attnLnW = WhisperOps.EnsureF32(w[$"{prefix}.attn_ln.weight"]);
        _attnLnB = WhisperOps.EnsureF32(w[$"{prefix}.attn_ln.bias"]);
        _qW = WhisperOps.EnsureF32(w[$"{prefix}.attn.query.weight"]); _qB = WhisperOps.EnsureF32(w[$"{prefix}.attn.query.bias"]);
        _kW = WhisperOps.EnsureF32(w[$"{prefix}.attn.key.weight"]);     // key has no bias
        _vW = WhisperOps.EnsureF32(w[$"{prefix}.attn.value.weight"]); _vB = WhisperOps.EnsureF32(w[$"{prefix}.attn.value.bias"]);
        _oW = WhisperOps.EnsureF32(w[$"{prefix}.attn.out.weight"]); _oB = WhisperOps.EnsureF32(w[$"{prefix}.attn.out.bias"]);
        _fsmnW = WhisperOps.EnsureF32(w[$"{prefix}.attn.fsmn_block.weight"]);   // [1280, 1, 31] depthwise
        _mlpLnW = WhisperOps.EnsureF32(w[$"{prefix}.mlp_ln.weight"]);
        _mlpLnB = WhisperOps.EnsureF32(w[$"{prefix}.mlp_ln.bias"]);
        _mlp0W = WhisperOps.EnsureF32(w[$"{prefix}.mlp.0.weight"]); _mlp0B = WhisperOps.EnsureF32(w[$"{prefix}.mlp.0.bias"]);
        _mlp2W = WhisperOps.EnsureF32(w[$"{prefix}.mlp.2.weight"]); _mlp2B = WhisperOps.EnsureF32(w[$"{prefix}.mlp.2.bias"]);
    }

    public Tensor Forward(IBackend backend, Tensor seq, int t, float[] cos, float[] sin)
    {
        // --- pre-norm FSMN attention ---
        Tensor normed = new(seq.Shape, DType.F32);
        backend.LayerNorm(normed, seq, _attnLnW!, _attnLnB!, 1e-5f);
        Tensor q = WhisperOps.ProjectLinear(backend, normed, _qW!, _qB, 1, t, NState, NState);
        Tensor k = WhisperOps.ProjectLinear(backend, normed, _kW!, null, 1, t, NState, NState);
        Tensor v = WhisperOps.ProjectLinear(backend, normed, _vW!, _vB, 1, t, NState, NState);
        normed.Dispose();

        Tensor fsm = Fsmn(backend, v, t);     // depthwise memory over value, [1, T, 1280]

        Tensor qH = new(new TensorShape(1, NHead, t, HeadDim), DType.F32);
        Tensor kH = new(new TensorShape(1, NHead, t, HeadDim), DType.F32);
        Tensor vH = new(new TensorShape(1, NHead, t, HeadDim), DType.F32);
        WhisperOps.ReshapeToMultiHead4D(qH, q, 1, t, NHead, HeadDim);
        WhisperOps.ReshapeToMultiHead4D(kH, k, 1, t, NHead, HeadDim);
        WhisperOps.ReshapeToMultiHead4D(vH, v, 1, t, NHead, HeadDim);
        q.Dispose(); k.Dispose(); v.Dispose();
        RopeHalf(qH, t, cos, sin);
        RopeHalf(kH, t, cos, sin);

        Tensor attn = new(new TensorShape(1, NHead, t, HeadDim), DType.F32);
        backend.ScaledDotProductAttention(attn, qH, kH, vH, null, 1f / MathF.Sqrt(HeadDim));
        qH.Dispose(); kH.Dispose(); vH.Dispose();

        Tensor wv = new(new TensorShape(1, t, NState), DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(wv, attn, 1, t, NHead, HeadDim);
        attn.Dispose();
        Tensor o = WhisperOps.ProjectLinear(backend, wv, _oW!, _oB, 1, t, NState, NState);
        wv.Dispose();
        AddInPlace(o, fsm);     // attention output = out(wv) + fsmn(value)
        fsm.Dispose();
        AddInPlace(o, seq);     // residual

        // --- pre-norm GELU MLP ---
        Tensor n2 = new(o.Shape, DType.F32);
        backend.LayerNorm(n2, o, _mlpLnW!, _mlpLnB!, 1e-5f);
        int ff = (int)_mlp0W!.Shape[0];
        Tensor f1 = WhisperOps.ProjectLinear(backend, n2, _mlp0W!, _mlp0B, 1, t, NState, ff);
        n2.Dispose();
        Activations.ErfGelu(f1);
        Tensor f2 = WhisperOps.ProjectLinear(backend, f1, _mlp2W!, _mlp2B, 1, t, ff, NState);
        f1.Dispose();
        AddInPlace(f2, o);
        o.Dispose();
        return f2;
    }

    /// <summary>FSMN memory branch: depthwise Conv1d (k31, 'same' pad 15/15, groups = channels) over the value
    /// stream plus a residual. Operates channels-last <c>[1, T, 1280]</c> via a transpose round-trip.</summary>
    private Tensor Fsmn(IBackend backend, Tensor v, int t)
    {
        Tensor vt = new(new TensorShape(1, NState, t), DType.F32);
        backend.Transpose2D(vt, v, t, NState);                 // [1,T,C] → [1,C,T]
        Tensor conv = new(new TensorShape(1, NState, t), DType.F32);
        int left = (FsmnKernel - 1) / 2, right = FsmnKernel - 1 - left;
        backend.Conv1d(conv, vt, _fsmnW!, null, 1, left, right, 1, NState);     // depthwise, groups=NState
        vt.Dispose();
        Tensor outT = new(new TensorShape(1, t, NState), DType.F32);
        backend.Transpose2D(outT, conv, NState, t);            // [1,C,T] → [1,T,C]
        conv.Dispose();
        AddInPlace(outT, v);                                   // + value (residual)
        return outT;
    }

    /// <summary>Rotate-half RoPE on a <c>[1, H, T, 64]</c> tensor in place: for pair (j, j+32),
    /// <c>x[j], x[j+32] ← x[j]·c - x[j+32]·s, x[j+32]·c + x[j]·s</c>, c/s indexed by (t, j).</summary>
    private static void RopeHalf(Tensor x, int t, float[] cos, float[] sin)
    {
        int half = HeadDim / 2;     // 32
        float* p = (float*)x.DataPointer;
        for (int h = 0; h < NHead; h++)
            for (int ti = 0; ti < t; ti++)
            {
                long baseI = (((long)h * t) + ti) * HeadDim;
                int cb = ti * half;
                for (int j = 0; j < half; j++)
                {
                    float c = cos[cb + j], s = sin[cb + j];
                    float a = p[baseI + j], b = p[baseI + j + half];
                    p[baseI + j] = a * c - b * s;
                    p[baseI + j + half] = b * c + a * s;
                }
            }
    }

    private static void AddInPlace(Tensor dst, Tensor src)
    {
        float* d = (float*)dst.DataPointer, s = (float*)src.DataPointer;
        for (long i = 0; i < dst.ElementCount; i++) d[i] += s[i];
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_attnLnW, _attnLnB, _qW, _qB, _kW, _vW, _vB, _oW, _oB, _fsmnW,
            _mlpLnW, _mlpLnB, _mlp0W, _mlp0B, _mlp2W, _mlp2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
