using HartsyInference.Audio.Models.Moonshine;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>S3Tokenizer (CosyVoice 2 variant) — converts a reference clip's mel into the discrete 25 Hz
/// FSQ speech-token stream the LM/flow consume. A supervised-ASR-style encoder (Whisper-class, a conv
/// stem that 4× subsamples 100 Hz → 25 Hz followed by 6 RoPE Transformer blocks) produces
/// semantically-aligned features that an 8-channel Finite Scalar Quantizer maps to <c>3^8 = 6561</c>
/// codes. Only the encode direction is needed at inference.
///
/// <para><b>Exact + tested:</b> <see cref="PackFsqTokens"/> implements the FSQ formula
/// (<c>tanh → round → shift → base-3 pack</c>) verbatim from the architecture doc. The encoder is the
/// real 6-block RoPE transformer stack: a 2× stride-2 Conv1d stem (kernel-3, 4× total subsample) →
/// <c>encoders.{i}</c> pre-norm self-attention + GELU FFN blocks (interleaved-RoPE rel-pos) →
/// <c>quantizer.project_down</c> to the 8 FSQ channels. The <c>speech_tokenizer_v2.onnx</c> file remains
/// available as an exact-weights fallback when the safetensors export is not present.</para></summary>
public sealed unsafe class S3Tokenizer : IDisposable
{
    private const int FsqDim = 8;
    private const int FsqLevels = 3;
    private const int NumEncoderBlocks = 6;
    private const int NumHeads = 8;
    private const float RopeTheta = 10_000f;
    private int _disposed;

    private Tensor?[] _subsampleW = new Tensor[2];
    private Tensor?[] _subsampleB = new Tensor[2];
    private readonly S3EncoderBlock[] _encoders = new S3EncoderBlock[NumEncoderBlocks];
    private Tensor? _afterNormW, _afterNormB;       // final encoder LayerNorm (optional)
    private Tensor? _projDownW, _projDownB;         // hidden → FsqDim

    public int VocabSize { get; } = 1;

    public S3Tokenizer()
    {
        VocabSize = 1;
        for (int i = 0; i < FsqDim; i++) VocabSize *= FsqLevels;     // 6561
        for (int i = 0; i < NumEncoderBlocks; i++) _encoders[i] = new S3EncoderBlock(NumHeads);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        for (int i = 0; i < 2; i++)
        {
            _subsampleW[i] = WhisperOps.EnsureF32(w[$"{p}subsample.{i}.weight"]);
            _subsampleB[i] = WhisperOps.EnsureF32(w[$"{p}subsample.{i}.bias"]);
        }
        for (int i = 0; i < NumEncoderBlocks; i++)
            _encoders[i].LoadWeights(w, $"{p}encoders.{i}");
        if (w.TryGetValue($"{p}after_norm.weight", out Tensor? anw))
        {
            _afterNormW = WhisperOps.EnsureF32(anw);
            _afterNormB = WhisperOps.EnsureF32(w[$"{p}after_norm.bias"]);
        }
        _projDownW = WhisperOps.EnsureF32(w[$"{p}quantizer.project_down.weight"]);
        _projDownB = w.TryGetValue($"{p}quantizer.project_down.bias", out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
    }

    /// <summary>Encodes a 100 Hz mel <c>[1, 80, T]</c> into 25 Hz speech-token IDs.</summary>
    public int[] Forward(IBackend backend, Tensor mel)
    {
        if (_projDownW is null) throw new InvalidOperationException("S3Tokenizer weights not loaded.");

        // 4× conv subsample (two stride-2 Conv1d) → ~25 Hz, channels-first [1, hidden, T].
        Tensor x = mel;
        bool owns = false;
        for (int i = 0; i < 2; i++)
        {
            Tensor wgt = _subsampleW![i]!;
            int outCh = (int)wgt.Shape[0];
            int k = (int)wgt.Shape[2];
            int inLen = (int)x.Shape[2];
            int pad = (k - 1) / 2;
            int outLen = (inLen + 2 * pad - (k - 1) - 1) / 2 + 1;
            Tensor nx = new(new TensorShape(1, outCh, outLen), DType.F32);
            backend.Conv1d(nx, x, wgt, _subsampleB![i], stride: 2, padLeft: pad, padRight: pad, dilation: 1, groups: 1);
            if (owns) x.Dispose();
            backend.LeakyRelu(nx, nx, 0f);
            x = nx;
            owns = true;
        }

        int hidden = (int)x.Shape[1];
        int t = (int)x.Shape[2];

        // To channels-last [1, T, hidden] for the transformer stack.
        Tensor seq = new(new TensorShape(1, t, hidden), DType.F32);
        backend.Transpose2D(seq, x, hidden, t);
        if (owns) x.Dispose();

        // RoPE tables shared by every block (interleaved convention).
        (float[] cos, float[] sin) = RotaryEmbedding.GetTables(hidden / NumHeads, RopeTheta, t);
        for (int i = 0; i < NumEncoderBlocks; i++)
        {
            Tensor next = _encoders[i].Forward(backend, seq, t, hidden, cos, sin);
            seq.Dispose();
            seq = next;
        }

        if (_afterNormW is not null)
        {
            Tensor normed = new(seq.Shape, DType.F32);
            backend.LayerNorm(normed, seq, _afterNormW!, _afterNormB!, 1e-5f);
            seq.Dispose();
            seq = normed;
        }

        // project_down to the 8 FSQ channels, then pack each frame.
        Tensor z = WhisperOps.ProjectLinear(backend, seq, _projDownW!, _projDownB, 1, t, hidden, FsqDim);
        seq.Dispose();
        int[] tokens = PackFsqTokens(z, FsqDim, FsqLevels);
        z.Dispose();
        return tokens;
    }

    /// <summary>FSQ tokenizer: for each frame, <c>token = Σ_j (round((L-1)/2·tanh(z_j)) + (L-1)/2)·L^j</c>.
    /// <paramref name="z"/> is the projected latent <c>[1, T, D]</c>; returns one base-L code per frame.</summary>
    public static int[] PackFsqTokens(Tensor z, int dim, int levels)
    {
        int t = (int)z.Shape[1];
        if ((int)z.Shape[2] != dim) throw new ArgumentException($"FSQ latent must have {dim} channels, got {z.Shape[2]}.");
        float half = (levels - 1) / 2f;
        float* zp = (float*)z.DataPointer;
        int[] tokens = new int[t];
        for (int frame = 0; frame < t; frame++)
        {
            int token = 0;
            int pow = 1;
            for (int j = 0; j < dim; j++)
            {
                float bounded = half * MathF.Tanh(zp[(long)frame * dim + j]);
                int zint = (int)MathF.Round(bounded);                       // {-(L-1)/2 .. +(L-1)/2}
                int shift = zint + (int)half;                               // {0 .. L-1}
                if (shift < 0) shift = 0; else if (shift > levels - 1) shift = levels - 1;
                token += shift * pow;
                pow *= levels;
            }
            tokens[frame] = token;
        }
        return tokens;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int i = 0; i < 2; i++)
        {
            if (_subsampleW[i] is not null) yield return _subsampleW[i]!;
            if (_subsampleB[i] is not null) yield return _subsampleB[i]!;
        }
        foreach (S3EncoderBlock blk in _encoders)
            foreach (Tensor t in blk.EnumerateWeights()) yield return t;
        if (_afterNormW is not null) yield return _afterNormW;
        if (_afterNormB is not null) yield return _afterNormB;
        if (_projDownW is not null) yield return _projDownW;
        if (_projDownB is not null) yield return _projDownB;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}

/// <summary>One S3 speech-tokenizer encoder block: pre-norm multi-head self-attention with interleaved
/// RoPE on the q/k heads, then a pre-norm position-wise GELU FFN, both with residual adds. Operates on a
/// channels-last <c>[1, T, hidden]</c> sequence. Mirrors the conformer-lite transformer layers used in
/// <c>s3tokenizer</c>'s encoder (no conv module in the v2 tokenizer's attention path).</summary>
internal sealed unsafe class S3EncoderBlock
{
    private readonly int _numHeads;
    private Tensor? _norm1W, _norm1B, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB;
    private Tensor? _norm2W, _norm2B, _ff1W, _ff1B, _ff2W, _ff2B;

    public S3EncoderBlock(int numHeads)
    {
        _numHeads = numHeads;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _norm1W = WhisperOps.EnsureF32(w[$"{prefix}.norm1.weight"]);
        _norm1B = WhisperOps.EnsureF32(w[$"{prefix}.norm1.bias"]);
        _qW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_q.weight"]); _qB = TryBias(w, $"{prefix}.self_attn.linear_q.bias");
        _kW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_k.weight"]); _kB = TryBias(w, $"{prefix}.self_attn.linear_k.bias");
        _vW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_v.weight"]); _vB = TryBias(w, $"{prefix}.self_attn.linear_v.bias");
        _oW = WhisperOps.EnsureF32(w[$"{prefix}.self_attn.linear_out.weight"]); _oB = TryBias(w, $"{prefix}.self_attn.linear_out.bias");
        _norm2W = WhisperOps.EnsureF32(w[$"{prefix}.norm2.weight"]);
        _norm2B = WhisperOps.EnsureF32(w[$"{prefix}.norm2.bias"]);
        _ff1W = WhisperOps.EnsureF32(w[$"{prefix}.feed_forward.w_1.weight"]); _ff1B = TryBias(w, $"{prefix}.feed_forward.w_1.bias");
        _ff2W = WhisperOps.EnsureF32(w[$"{prefix}.feed_forward.w_2.weight"]); _ff2B = TryBias(w, $"{prefix}.feed_forward.w_2.bias");
    }

    private static Tensor? TryBias(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? b) ? WhisperOps.EnsureF32(b) : null;

    /// <summary>Forwards a <c>[1, T, hidden]</c> sequence through self-attention + FFN, returning a new
    /// <c>[1, T, hidden]</c> tensor (the caller owns and disposes both input and output).</summary>
    public Tensor Forward(IBackend backend, Tensor seq, int t, int hidden, float[] cos, float[] sin)
    {
        int headDim = hidden / _numHeads;

        Tensor normed = new(seq.Shape, DType.F32);
        backend.LayerNorm(normed, seq, _norm1W!, _norm1B!, 1e-5f);
        Tensor q = WhisperOps.ProjectLinear(backend, normed, _qW!, _qB, 1, t, hidden, hidden);
        Tensor k = WhisperOps.ProjectLinear(backend, normed, _kW!, _kB, 1, t, hidden, hidden);
        Tensor v = WhisperOps.ProjectLinear(backend, normed, _vW!, _vB, 1, t, hidden, hidden);
        normed.Dispose();

        // [1, T, H*D] → [1, H, T, D].
        Tensor qH = new(new TensorShape(1, _numHeads, t, headDim), DType.F32);
        Tensor kH = new(new TensorShape(1, _numHeads, t, headDim), DType.F32);
        Tensor vH = new(new TensorShape(1, _numHeads, t, headDim), DType.F32);
        WhisperOps.ReshapeToMultiHead4D(qH, q, 1, t, _numHeads, headDim);
        WhisperOps.ReshapeToMultiHead4D(kH, k, 1, t, _numHeads, headDim);
        WhisperOps.ReshapeToMultiHead4D(vH, v, 1, t, _numHeads, headDim);
        q.Dispose(); k.Dispose(); v.Dispose();

        RotaryEmbedding.ApplyInPlace(qH, _numHeads, t, headDim, headDim, 0, cos, sin);
        RotaryEmbedding.ApplyInPlace(kH, _numHeads, t, headDim, headDim, 0, cos, sin);

        Tensor attn = new(new TensorShape(1, _numHeads, t, headDim), DType.F32);
        backend.ScaledDotProductAttention(attn, qH, kH, vH, mask: null, 1f / MathF.Sqrt(headDim));
        qH.Dispose(); kH.Dispose(); vH.Dispose();

        Tensor merged = new(new TensorShape(1, t, hidden), DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(merged, attn, 1, t, _numHeads, headDim);
        attn.Dispose();
        Tensor o = WhisperOps.ProjectLinear(backend, merged, _oW!, _oB, 1, t, hidden, hidden);
        merged.Dispose();
        AddInPlace(o, seq);                                        // residual

        Tensor n2 = new(o.Shape, DType.F32);
        backend.LayerNorm(n2, o, _norm2W!, _norm2B!, 1e-5f);
        int ffDim = (int)_ff1W!.Shape[0];
        Tensor f1 = WhisperOps.ProjectLinear(backend, n2, _ff1W!, _ff1B, 1, t, hidden, ffDim);
        n2.Dispose();
        backend.Gelu(f1, f1);
        Tensor f2 = WhisperOps.ProjectLinear(backend, f1, _ff2W!, _ff2B, 1, t, ffDim, hidden);
        f1.Dispose();
        AddInPlace(f2, o);
        o.Dispose();
        return f2;
    }

    private static void AddInPlace(Tensor dst, Tensor src)
    {
        float* dp = (float*)dst.DataPointer;
        float* sp = (float*)src.DataPointer;
        long n = dst.ElementCount;
        for (long i = 0; i < n; i++) dp[i] += sp[i];
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_norm1W, _norm1B, _qW, _qB, _kW, _kB, _vW, _vB, _oW, _oB, _norm2W, _norm2B, _ff1W, _ff1B, _ff2W, _ff2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
