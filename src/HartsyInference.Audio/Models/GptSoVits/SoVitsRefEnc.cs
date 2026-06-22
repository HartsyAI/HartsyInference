using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.GptSoVits;

/// <summary>GPT-SoVITS v2Pro reference encoder (<c>ref_enc</c>, a StyleSpeech <b>MelStyleEncoder</b>): turns a
/// reference mel/spec <c>[1, 704, T]</c> into the broadcast speaker embedding <c>ge [1, 512, 1]</c>. Path:
/// <c>spectral</c> (Linear→Mish→Linear→Mish, per frame) → 2× <c>Conv1dGLU</c> (depthwise-ish gated conv over
/// time) → masked multi-head <c>slf_attn</c> (n_head 2) → <c>fc</c> Linear(→512) → masked average-pool over
/// time. All inference here is unmasked (single utterance, no padding) so the "masked" pool is a plain mean.
///
/// <para><b>Reuse:</b> linear projections via <see cref="WhisperOps.ProjectLinear"/>, attention via
/// <see cref="IBackend.ScaledDotProductAttention"/>. Net-new: Mish, the Conv1dGLU gate, and the time pool —
/// all small pointer walks.</para></summary>
public sealed unsafe class SoVitsRefEnc
{
    private readonly int _nMels, _hidden, _styleDim, _nHead;
    private Tensor? _spec1W, _spec1B, _spec2W, _spec2B;
    private Tensor? _glu1W, _glu1B, _glu2W, _glu2B;
    private Tensor? _attnQkvW, _attnQkvB, _attnOW, _attnOB;
    private Tensor? _attnQW, _attnQB, _attnKW, _attnKB, _attnVW, _attnVB;
    private bool _fusedQkv;
    private Tensor? _fcW, _fcB;

    public SoVitsRefEnc(int nMels = 704, int hidden = 128, int styleDim = 512, int nHead = 2, int kernel = 5)
    {
        _nMels = nMels; _hidden = hidden; _styleDim = styleDim; _nHead = nHead;
        _ = kernel;     // kernel is read from the conv weight shape at GLU time; param kept for the call contract
    }

    public int StyleDim => _styleDim;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "ref_enc")
    {
        _spec1W = Lin(w, $"{prefix}.spectral.0"); _spec1B = WhisperOps.EnsureF32(w[$"{prefix}.spectral.0.bias"]);
        _spec2W = Lin(w, $"{prefix}.spectral.3"); _spec2B = WhisperOps.EnsureF32(w[$"{prefix}.spectral.3.bias"]);
        _glu1W = WhisperOps.EnsureF32(w[$"{prefix}.temporal.0.conv1.conv.weight"]);
        _glu1B = w.TryGetValue($"{prefix}.temporal.0.conv1.conv.bias", out Tensor? g1b) ? WhisperOps.EnsureF32(g1b) : null;
        _glu2W = WhisperOps.EnsureF32(w[$"{prefix}.temporal.1.conv1.conv.weight"]);
        _glu2B = w.TryGetValue($"{prefix}.temporal.1.conv1.conv.bias", out Tensor? g2b) ? WhisperOps.EnsureF32(g2b) : null;
        if (w.ContainsKey($"{prefix}.slf_attn.qkv.weight") || w.ContainsKey($"{prefix}.slf_attn.in_proj_weight"))
        {
            _fusedQkv = true;
            string qkvKey = w.ContainsKey($"{prefix}.slf_attn.qkv.weight") ? $"{prefix}.slf_attn.qkv" : $"{prefix}.slf_attn.in_proj";
            _attnQkvW = WhisperOps.EnsureF32(w[$"{qkvKey}.weight"]);
            _attnQkvB = w.TryGetValue($"{qkvKey}.bias", out Tensor? qkvb) ? WhisperOps.EnsureF32(qkvb) : null;
        }
        else
        {
            _attnQW = Lin(w, $"{prefix}.slf_attn.w_qs"); _attnQB = Bias(w, $"{prefix}.slf_attn.w_qs");
            _attnKW = Lin(w, $"{prefix}.slf_attn.w_ks"); _attnKB = Bias(w, $"{prefix}.slf_attn.w_ks");
            _attnVW = Lin(w, $"{prefix}.slf_attn.w_vs"); _attnVB = Bias(w, $"{prefix}.slf_attn.w_vs");
        }
        _attnOW = Lin(w, $"{prefix}.slf_attn.fc"); _attnOB = Bias(w, $"{prefix}.slf_attn.fc");
        _fcW = Lin(w, $"{prefix}.fc"); _fcB = Bias(w, $"{prefix}.fc");
    }

    /// <summary>Encodes a reference spec <c>[1, nMels, t]</c> → speaker embedding <c>ge [1, styleDim, 1]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor refSpec, int t)
    {
        // spectral: operate per-frame as [1, t, nMels] (Linear over channel dim).
        Tensor seq = ToSeq(refSpec, _nMels, t);                                     // [1, t, nMels]
        Tensor s1 = WhisperOps.ProjectLinear(backend, seq, _spec1W!, _spec1B, 1, t, _nMels, _hidden); seq.Dispose();
        Mish(s1);
        Tensor s2 = WhisperOps.ProjectLinear(backend, s1, _spec2W!, _spec2B, 1, t, _hidden, _hidden); s1.Dispose();
        Mish(s2);

        // temporal: 2× Conv1dGLU on [1, hidden, t]; GLU doubles channels then gates.
        Tensor chFirst = ToChannelsFirst(s2, _hidden, t); s2.Dispose();             // [1, hidden, t]
        Tensor g1 = Conv1dGlu(backend, chFirst, _glu1W!, _glu1B, t); chFirst.Dispose();
        Tensor g2 = Conv1dGlu(backend, g1, _glu2W!, _glu2B, t); g1.Dispose();

        // slf_attn: multi-head self-attention over time on [1, t, hidden].
        Tensor attnIn = ToSeq(g2, _hidden, t); g2.Dispose();                        // [1, t, hidden]
        Tensor attnOut = SelfAttention(backend, attnIn, t); attnIn.Dispose();        // [1, t, hidden]

        // fc → style, then average-pool over time → [1, styleDim, 1].
        Tensor style = WhisperOps.ProjectLinear(backend, attnOut, _fcW!, _fcB, 1, t, _hidden, _styleDim);
        attnOut.Dispose();
        Tensor ge = new(new TensorShape(1, _styleDim, 1), DType.F32);
        float* stp = (float*)style.DataPointer;
        float* gep = (float*)ge.DataPointer;
        float invT = 1f / t;
        for (int d = 0; d < _styleDim; d++)
        {
            float acc = 0f;
            for (int j = 0; j < t; j++) acc += stp[(long)j * _styleDim + d];
            gep[d] = acc * invT;
        }
        style.Dispose();
        return ge;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own =
        [
            _spec1W, _spec1B, _spec2W, _spec2B, _glu1W, _glu1B, _glu2W, _glu2B,
            _attnQkvW, _attnQkvB, _attnQW, _attnQB, _attnKW, _attnKB, _attnVW, _attnVB,
            _attnOW, _attnOB, _fcW, _fcB,
        ];
        foreach (Tensor? t in own) if (t is not null) yield return t;
    }

    private Tensor SelfAttention(IBackend backend, Tensor x, int t)
    {
        int hd = _hidden / _nHead;
        float scale = 1f / MathF.Sqrt(hd);
        Tensor q, k, v;
        if (_fusedQkv)
        {
            Tensor qkv = WhisperOps.ProjectLinear(backend, x, _attnQkvW!, _attnQkvB, 1, t, _hidden, 3 * _hidden);
            (q, k, v) = SplitQkv(qkv, t); qkv.Dispose();
        }
        else
        {
            q = WhisperOps.ProjectLinear(backend, x, _attnQW!, _attnQB, 1, t, _hidden, _hidden);
            k = WhisperOps.ProjectLinear(backend, x, _attnKW!, _attnKB, 1, t, _hidden, _hidden);
            v = WhisperOps.ProjectLinear(backend, x, _attnVW!, _attnVB, 1, t, _hidden, _hidden);
        }
        Tensor q4 = new(new TensorShape(1, _nHead, t, hd), DType.F32);
        Tensor k4 = new(new TensorShape(1, _nHead, t, hd), DType.F32);
        Tensor v4 = new(new TensorShape(1, _nHead, t, hd), DType.F32);
        WhisperOps.ReshapeToMultiHead4D(q4, q, 1, t, _nHead, hd);
        WhisperOps.ReshapeToMultiHead4D(k4, k, 1, t, _nHead, hd);
        WhisperOps.ReshapeToMultiHead4D(v4, v, 1, t, _nHead, hd);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor ctx4 = new(new TensorShape(1, _nHead, t, hd), DType.F32);
        backend.ScaledDotProductAttention(ctx4, q4, k4, v4, null, scale);
        q4.Dispose(); k4.Dispose(); v4.Dispose();
        Tensor ctx = new(new TensorShape(1, t, _hidden), DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(ctx, ctx4, 1, t, _nHead, hd); ctx4.Dispose();
        Tensor outT = WhisperOps.ProjectLinear(backend, ctx, _attnOW!, _attnOB, 1, t, _hidden, _hidden); ctx.Dispose();
        return outT;
    }

    private (Tensor Q, Tensor K, Tensor V) SplitQkv(Tensor qkv, int t)
    {
        Tensor q = new(new TensorShape(1, t, _hidden), DType.F32);
        Tensor k = new(new TensorShape(1, t, _hidden), DType.F32);
        Tensor v = new(new TensorShape(1, t, _hidden), DType.F32);
        float* src = (float*)qkv.DataPointer;
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
        for (int j = 0; j < t; j++)
        {
            long row = (long)j * 3 * _hidden;
            Buffer.MemoryCopy(src + row, qp + (long)j * _hidden, _hidden * 4, _hidden * 4);
            Buffer.MemoryCopy(src + row + _hidden, kp + (long)j * _hidden, _hidden * 4, _hidden * 4);
            Buffer.MemoryCopy(src + row + 2 * _hidden, vp + (long)j * _hidden, _hidden * 4, _hidden * 4);
        }
        return (q, k, v);
    }

    /// <summary>Conv1dGLU: conv1 maps <c>hidden → 2·hidden</c> (same-pad), then split halves <c>a, b</c> and
    /// gate <c>a · sigmoid(b)</c>, with a residual add (StyleSpeech). Input/output <c>[1, hidden, t]</c>.</summary>
    private Tensor Conv1dGlu(IBackend backend, Tensor x, Tensor convW, Tensor? convB, int t)
    {
        int kernel = (int)convW.Shape[2];
        int pad = (kernel - 1) / 2;
        Tensor proj = new(new TensorShape(1, 2 * _hidden, t), DType.F32);
        backend.Conv1d(proj, x, convW, convB, 1, pad, pad, 1, 1);
        Tensor outT = new(new TensorShape(1, _hidden, t), DType.F32);
        float* pp = (float*)proj.DataPointer;
        float* xp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int c = 0; c < _hidden; c++)
            for (int j = 0; j < t; j++)
            {
                float a = pp[(long)c * t + j];
                float b = pp[(long)(_hidden + c) * t + j];
                float gated = a * (1f / (1f + MathF.Exp(-b)));
                op[(long)c * t + j] = gated + xp[(long)c * t + j];
            }
        proj.Dispose();
        return outT;
    }

    private static Tensor ToSeq(Tensor chFirst, int channels, int t)
    {
        Tensor outT = new(new TensorShape(1, t, channels), DType.F32);
        float* src = (float*)chFirst.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int c = 0; c < channels; c++)
            for (int j = 0; j < t; j++) dst[(long)j * channels + c] = src[(long)c * t + j];
        return outT;
    }

    private static Tensor ToChannelsFirst(Tensor seq, int channels, int t)
    {
        Tensor outT = new(new TensorShape(1, channels, t), DType.F32);
        float* src = (float*)seq.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int j = 0; j < t; j++)
            for (int c = 0; c < channels; c++) dst[(long)c * t + j] = src[(long)j * channels + c];
        return outT;
    }

    private static void Mish(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        for (long i = 0; i < n; i++)
        {
            float x = p[i];
            p[i] = x * MathF.Tanh(MathF.Log(1f + MathF.Exp(x)));
        }
    }

    private static Tensor Lin(IReadOnlyDictionary<string, Tensor> w, string key)
        => WhisperOps.EnsureF32(w[$"{key}.weight"]);

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue($"{key}.bias", out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
}
