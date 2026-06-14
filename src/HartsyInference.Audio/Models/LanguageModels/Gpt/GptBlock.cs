using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.LanguageModels.Gpt;

/// <summary>One GPT-2 pre-norm block: <c>x + Attn(LN1(x))</c> then <c>x + MLP(LN2(x))</c>. Fused QKV
/// projection, multi-head SDPA, 4× GELU MLP, all linears <c>bias=False</c>; LayerNorms carry weight+bias.
/// Key scheme follows HF Bark (<c>layernorm_1</c>, <c>attn.att_proj</c>, <c>attn.out_proj</c>,
/// <c>layernorm_2</c>, <c>mlp.in_proj</c>, <c>mlp.out_proj</c>).</summary>
public sealed unsafe class GptBlock : IDisposable
{
    private readonly GptConfig _cfg;
    private int _disposed;

    private Tensor? _ln1G, _ln1B, _ln2G, _ln2B;
    private Tensor? _attW, _outW;         // att_proj [3H, H], out_proj [H, H]
    private Tensor? _mlpInW, _mlpOutW;    // [4H, H], [H, 4H]

    public GptBlock(GptConfig cfg) => _cfg = cfg;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _ln1G = WhisperOps.EnsureF32(w[$"{prefix}.layernorm_1.weight"]);
        _ln1B = WhisperOps.EnsureF32(w[$"{prefix}.layernorm_1.bias"]);
        _attW = WhisperOps.EnsureF32(w[$"{prefix}.attn.att_proj.weight"]);
        _outW = WhisperOps.EnsureF32(w[$"{prefix}.attn.out_proj.weight"]);
        _ln2G = WhisperOps.EnsureF32(w[$"{prefix}.layernorm_2.weight"]);
        _ln2B = WhisperOps.EnsureF32(w[$"{prefix}.layernorm_2.bias"]);
        _mlpInW = WhisperOps.EnsureF32(w[$"{prefix}.mlp.in_proj.weight"]);
        _mlpOutW = WhisperOps.EnsureF32(w[$"{prefix}.mlp.out_proj.weight"]);
    }

    public Tensor Forward(IBackend backend, Tensor x, Tensor? causalMask)
    {
        int t = (int)x.Shape[1];
        int h = _cfg.Hidden;
        int nh = _cfg.NumHeads;
        int d = _cfg.HeadDim;

        // ── Attention ──
        Tensor ln1 = new(x.Shape, DType.F32);
        backend.LayerNorm(ln1, x, _ln1G!, _ln1B!, 1e-5f);
        Tensor qkv = WhisperOps.ProjectLinear(backend, ln1, _attW!, bias: null, 1, t, h, 3 * h);
        ln1.Dispose();

        (Tensor q, Tensor k, Tensor v) = SplitQkv(qkv, t, h);
        qkv.Dispose();
        Tensor qMh = ToHeads(q, t, nh, d); q.Dispose();
        Tensor kMh = ToHeads(k, t, nh, d); k.Dispose();
        Tensor vMh = ToHeads(v, t, nh, d); v.Dispose();

        Tensor attn = new(new TensorShape(1, nh, t, d), DType.F32);
        backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, causalMask, 1f / MathF.Sqrt(d));
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor attnFlat = new(new TensorShape(1, t, h), DType.F32);
        WhisperOps.ReshapeFromMultiHead4D(attnFlat, attn, 1, t, nh, d);
        attn.Dispose();
        Tensor attnOut = WhisperOps.ProjectLinear(backend, attnFlat, _outW!, bias: null, 1, t, h, h);
        attnFlat.Dispose();

        Tensor res1 = new(x.Shape, DType.F32);
        AddInto(res1, x, attnOut);
        attnOut.Dispose();

        // ── MLP ──
        Tensor ln2 = new(res1.Shape, DType.F32);
        backend.LayerNorm(ln2, res1, _ln2G!, _ln2B!, 1e-5f);
        Tensor fc = WhisperOps.ProjectLinear(backend, ln2, _mlpInW!, bias: null, 1, t, h, _cfg.MlpDim);
        ln2.Dispose();
        backend.Gelu(fc, fc);
        Tensor proj = WhisperOps.ProjectLinear(backend, fc, _mlpOutW!, bias: null, 1, t, _cfg.MlpDim, h);
        fc.Dispose();

        Tensor res2 = new(res1.Shape, DType.F32);
        AddInto(res2, res1, proj);
        res1.Dispose();
        proj.Dispose();
        return res2;
    }

    private static (Tensor q, Tensor k, Tensor v) SplitQkv(Tensor qkv, int t, int h)
    {
        Tensor q = new(new TensorShape(1, t, h), DType.F32);
        Tensor k = new(new TensorShape(1, t, h), DType.F32);
        Tensor v = new(new TensorShape(1, t, h), DType.F32);
        float* src = (float*)qkv.DataPointer;
        float* qp = (float*)q.DataPointer, kp = (float*)k.DataPointer, vp = (float*)v.DataPointer;
        for (int s = 0; s < t; s++)
        {
            long row = (long)s * 3 * h;
            long dst = (long)s * h;
            for (int c = 0; c < h; c++)
            {
                qp[dst + c] = src[row + c];
                kp[dst + c] = src[row + h + c];
                vp[dst + c] = src[row + 2 * h + c];
            }
        }
        return (q, k, v);
    }

    private static Tensor ToHeads(Tensor seq, int t, int nh, int d)
    {
        Tensor outT = new(new TensorShape(1, nh, t, d), DType.F32);
        WhisperOps.ReshapeToMultiHead4D(outT, seq, 1, t, nh, d);
        return outT;
    }

    private static void AddInto(Tensor dst, Tensor a, Tensor b)
    {
        float* dp = (float*)dst.DataPointer;
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        long n = dst.ElementCount;
        for (long i = 0; i < n; i++) dp[i] = ap[i] + bp[i];
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_ln1G, _ln1B, _attW, _outW, _ln2G, _ln2B, _mlpInW, _mlpOutW];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
