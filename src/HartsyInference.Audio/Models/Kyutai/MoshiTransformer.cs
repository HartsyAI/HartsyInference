using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Kyutai;

/// <summary>Moshi/Kyutai temporal backbone ("Helium" <c>StreamingTransformer</c>), the 16-layer decoder that
/// drives the TTS depth transformer. Each layer is <c>x += selfAttn(rms(x)); x += crossAttn(ln(x), cond);
/// x += gating(rms(x))</c> with: fused QKV (<c>self_attn.in_proj_weight</c> [3·dim, dim]), interleaved RoPE
/// (θ 10000), causal SDPA; a speaker cross-attention sublayer (Q from the stream, K/V from the conditioning,
/// also RoPE'd) gated by a real LayerNorm <c>norm_cross</c>; and a SwiGLU <c>gating</c> (linear_in fuses
/// gate+up at inner = ⅔·dim_feedforward = 5632, silu(gate)·up, linear_out). Norms are RMSNorm with the weight
/// stored as <c>alpha</c> (eps 1e-5). A final <c>out_norm</c> RMSNorm precedes the text head.
///
/// <para>Validated numerically against the real <c>kyutai/tts-1.6b-en_fr</c> backbone (see
/// <c>KyutaiBackboneParityTests</c>). TODO(gpu-residency): the head-split / weight-slice helpers below loop on
/// host pointers; fold them into backend kernels so a CUDA run stays device-resident.</para></summary>
public sealed unsafe class MoshiTransformer : IDisposable
{
    public const int Dim = 2048, Heads = 16, HeadDim = 128, GateInner = 5632;
    private const float RmsEps = 1e-5f, LnEps = 1e-5f, RopeTheta = 10_000f;

    private readonly int _layers;
    private readonly Tensor?[] _selfIn, _selfOut, _crossQ, _crossKV, _crossOut, _gateIn, _gateOut;
    private readonly Tensor?[] _norm1, _norm2, _normCrossW, _normCrossB;
    private Tensor? _outNorm;
    private int _disposed;

    public MoshiTransformer(int layers = 16)
    {
        _layers = layers;
        _selfIn = new Tensor?[layers]; _selfOut = new Tensor?[layers];
        _crossQ = new Tensor?[layers]; _crossKV = new Tensor?[layers]; _crossOut = new Tensor?[layers];
        _gateIn = new Tensor?[layers]; _gateOut = new Tensor?[layers];
        _norm1 = new Tensor?[layers]; _norm2 = new Tensor?[layers];
        _normCrossW = new Tensor?[layers]; _normCrossB = new Tensor?[layers];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "transformer")
    {
        for (int i = 0; i < _layers; i++)
        {
            string p = $"{prefix}.layers.{i}";
            _selfIn[i] = WhisperOps.EnsureF32(w[$"{p}.self_attn.in_proj_weight"]);     // [3·dim, dim]
            _selfOut[i] = WhisperOps.EnsureF32(w[$"{p}.self_attn.out_proj.weight"]);   // [dim, dim]
            Tensor crossIn = WhisperOps.EnsureF32(w[$"{p}.cross_attention.in_proj_weight"]); // [3·dim, dim]
            _crossQ[i] = SliceRows(crossIn, 0, Dim);          // Q rows → [dim, dim]
            _crossKV[i] = SliceRows(crossIn, Dim, 3 * Dim);   // K,V rows → [2·dim, dim]
            _crossOut[i] = WhisperOps.EnsureF32(w[$"{p}.cross_attention.out_proj.weight"]);
            _norm1[i] = FlattenAlpha(WhisperOps.EnsureF32(w[$"{p}.norm1.alpha"]));
            _norm2[i] = FlattenAlpha(WhisperOps.EnsureF32(w[$"{p}.norm2.alpha"]));
            _normCrossW[i] = WhisperOps.EnsureF32(w[$"{p}.norm_cross.weight"]);
            _normCrossB[i] = WhisperOps.EnsureF32(w[$"{p}.norm_cross.bias"]);
            _gateIn[i] = WhisperOps.EnsureF32(w[$"{p}.gating.linear_in.weight"]);      // [2·inner, dim]
            _gateOut[i] = WhisperOps.EnsureF32(w[$"{p}.gating.linear_out.weight"]);    // [dim, inner]
        }
        _outNorm = FlattenAlpha(WhisperOps.EnsureF32(w[$"out_norm.alpha"]));
    }

    /// <summary>Runs the full backbone on a precomputed input embedding <paramref name="input"/> <c>[1,T,dim]</c>
    /// with cross-attention conditioning <paramref name="cross"/> <c>[1,S,dim]</c>. Returns the post-out_norm
    /// hidden state <c>[1,T,dim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor input, Tensor cross)
    {
        int t = (int)input.Shape[1], s = (int)cross.Shape[1];
        (Tensor cosT, Tensor sinT) = BuildRope(t);
        Tensor causal = WhisperOps.BuildCausalMask(t);

        Tensor x = new(new TensorShape(1, t, Dim), DType.F32);
        Buffer.MemoryCopy((void*)input.DataPointer, (void*)x.DataPointer, (long)t * Dim * 4, (long)t * Dim * 4);

        for (int i = 0; i < _layers; i++)
        {
            x = SelfAttn(backend, x, i, t, cosT, sinT, causal);
            x = CrossAttn(backend, x, cross, i, t, s);
            x = Gating(backend, x, i, t);
        }

        Tensor outT = new(new TensorShape(1, t, Dim), DType.F32);
        backend.RmsNorm(outT, x, _outNorm!, RmsEps);
        x.Dispose();
        cosT.Dispose(); sinT.Dispose(); causal.Dispose();
        return outT;
    }

    private Tensor SelfAttn(IBackend backend, Tensor x, int layer, int t, Tensor cos, Tensor sin, Tensor causal)
    {
        Tensor pre = new(new TensorShape(1, t, Dim), DType.F32);
        backend.RmsNorm(pre, x, _norm1[layer]!, RmsEps);
        Tensor qkv = WhisperOps.ProjectLinear(backend, pre, _selfIn[layer]!, null, 1, t, Dim, 3 * Dim);
        pre.Dispose();
        (Tensor q, Tensor k, Tensor v) = SplitQkv(qkv, t); qkv.Dispose();

        backend.ApplyRopeInterleaved(q, cos, sin);
        backend.ApplyRopeInterleaved(k, cos, sin);
        Tensor attn = Attend(backend, q, k, v, t, t, causal);
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor o = WhisperOps.ProjectLinear(backend, attn, _selfOut[layer]!, null, 1, t, Dim, Dim);
        attn.Dispose();
        backend.Add(o, o, x);   // residual
        x.Dispose();
        return o;
    }

    private Tensor CrossAttn(IBackend backend, Tensor x, Tensor cross, int layer, int t, int s)
    {
        Tensor pre = new(new TensorShape(1, t, Dim), DType.F32);
        backend.LayerNorm(pre, x, _normCrossW[layer]!, _normCrossB[layer]!, LnEps);
        Tensor qFlat = WhisperOps.ProjectLinear(backend, pre, _crossQ[layer]!, null, 1, t, Dim, Dim);
        pre.Dispose();
        Tensor kvFlat = WhisperOps.ProjectLinear(backend, cross, _crossKV[layer]!, null, 1, s, Dim, 2 * Dim);

        // Cross-attention is RoPE-free (moshi: "rope and cross_attention makes no sense").
        Tensor q = ToHeads(qFlat, t); qFlat.Dispose();
        (Tensor k, Tensor v) = SplitKv(kvFlat, s); kvFlat.Dispose();
        Tensor attn = Attend(backend, q, k, v, t, s, null);   // full cross attention, no mask
        q.Dispose(); k.Dispose(); v.Dispose();

        Tensor o = WhisperOps.ProjectLinear(backend, attn, _crossOut[layer]!, null, 1, t, Dim, Dim);
        attn.Dispose();
        backend.Add(o, o, x);
        x.Dispose();
        return o;
    }

    private Tensor Gating(IBackend backend, Tensor x, int layer, int t)
    {
        Tensor pre = new(new TensorShape(1, t, Dim), DType.F32);
        backend.RmsNorm(pre, x, _norm2[layer]!, RmsEps);
        Tensor gu = WhisperOps.ProjectLinear(backend, pre, _gateIn[layer]!, null, 1, t, Dim, 2 * GateInner);
        pre.Dispose();

        // silu(gate) * up where gate = gu[..., :inner], up = gu[..., inner:].
        Tensor act = new(new TensorShape(1, t, GateInner), DType.F32);
        float* g = (float*)gu.DataPointer; float* a = (float*)act.DataPointer;
        for (int r = 0; r < t; r++)
        {
            float* row = g + (long)r * 2 * GateInner;
            float* outRow = a + (long)r * GateInner;
            for (int c = 0; c < GateInner; c++)
            {
                float gate = row[c];
                outRow[c] = (gate / (1f + MathF.Exp(-gate))) * row[GateInner + c];
            }
        }
        gu.Dispose();
        Tensor o = WhisperOps.ProjectLinear(backend, act, _gateOut[layer]!, null, 1, t, GateInner, Dim);
        act.Dispose();
        backend.Add(o, o, x);
        x.Dispose();
        return o;
    }

    /// <summary>SDPA for q <c>[1,T,H,D]</c> against k/v <c>[1,Tk,H,D]</c>: permute to <c>[1,H,T,D]</c>, attend,
    /// flatten back to <c>[1,T,dim]</c>.</summary>
    private static Tensor Attend(IBackend backend, Tensor q, Tensor k, Tensor v, int tq, int tk, Tensor? mask)
    {
        Tensor qMh = new(new TensorShape(1, Heads, tq, HeadDim), DType.F32);
        Tensor kMh = new(new TensorShape(1, Heads, tk, HeadDim), DType.F32);
        Tensor vMh = new(new TensorShape(1, Heads, tk, HeadDim), DType.F32);
        backend.Permute0213(qMh, q, tq, Heads, HeadDim);
        backend.Permute0213(kMh, k, tk, Heads, HeadDim);
        backend.Permute0213(vMh, v, tk, Heads, HeadDim);

        Tensor attn = new(new TensorShape(1, Heads, tq, HeadDim), DType.F32);
        backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, mask, 1f / MathF.Sqrt(HeadDim));
        qMh.Dispose(); kMh.Dispose(); vMh.Dispose();

        Tensor flat = new(new TensorShape(1, tq, Dim), DType.F32);
        float* ap = (float*)attn.DataPointer; float* fp = (float*)flat.DataPointer;
        for (int h = 0; h < Heads; h++)
            for (int tt = 0; tt < tq; tt++)
                Buffer.MemoryCopy(ap + (((long)h * tq + tt) * HeadDim), fp + (((long)tt * Heads + h) * HeadDim), HeadDim * 4, HeadDim * 4);
        attn.Dispose();
        return flat;
    }

    // [1,T,3·dim] → q,k,v each [1,T,H,D].
    private static (Tensor, Tensor, Tensor) SplitQkv(Tensor qkv, int t)
    {
        Tensor q = ColumnSlice(qkv, t, 0), k = ColumnSlice(qkv, t, Dim), v = ColumnSlice(qkv, t, 2 * Dim);
        return (q, k, v);
    }

    // [1,S,2·dim] → k,v each [1,S,H,D].
    private static (Tensor, Tensor) SplitKv(Tensor kv, int s)
        => (ColumnSlice(kv, s, 0), ColumnSlice(kv, s, Dim));

    // [1,T,dim] → [1,T,H,D] (contiguous, just a reshape via copy).
    private static Tensor ToHeads(Tensor flat, int t) => ColumnSlice(flat, t, 0);

    // Copies `Dim` columns starting at `colOff` of a [1,rows,stride] tensor into a fresh [1,rows,H,D] tensor.
    private static Tensor ColumnSlice(Tensor src, int rows, int colOff)
    {
        int stride = (int)src.Shape[src.Shape.Rank - 1];
        Tensor outT = new(new TensorShape(1, rows, Heads, HeadDim), DType.F32);
        float* sp = (float*)src.DataPointer; float* op = (float*)outT.DataPointer;
        for (int r = 0; r < rows; r++)
            Buffer.MemoryCopy(sp + (long)r * stride + colOff, op + (long)r * Dim, Dim * 4, Dim * 4);
        return outT;
    }

    // Copy rows [r0,r1) of a [outDim, inDim] weight into a fresh [r1-r0, inDim] weight.
    private static Tensor SliceRows(Tensor w, int r0, int r1)
    {
        int inDim = (int)w.Shape[1];
        Tensor outT = new(new TensorShape(r1 - r0, inDim), DType.F32);
        Buffer.MemoryCopy((float*)w.DataPointer + (long)r0 * inDim, (void*)outT.DataPointer,
            (long)(r1 - r0) * inDim * 4, (long)(r1 - r0) * inDim * 4);
        return outT;
    }

    // [1,1,dim] alpha → [dim] (contiguous reinterpret via copy) for RmsNorm.
    private static Tensor FlattenAlpha(Tensor alpha)
    {
        Tensor outT = new(new TensorShape(Dim), DType.F32);
        Buffer.MemoryCopy((void*)alpha.DataPointer, (void*)outT.DataPointer, (long)Dim * 4, (long)Dim * 4);
        return outT;
    }

    private static (Tensor, Tensor) BuildRope(int t)
    {
        (double[] invFreq, double mscale) = RopeFrequencyBuilder.Build(HeadDim, RopeTheta, null, t);
        int half = HeadDim / 2;
        Tensor cos = new(new TensorShape(1, t, HeadDim), DType.F32);
        Tensor sin = new(new TensorShape(1, t, HeadDim), DType.F32);
        float* pc = (float*)cos.DataPointer; float* ps = (float*)sin.DataPointer;
        for (int s = 0; s < t; s++)
            for (int i = 0; i < half; i++)
            {
                double angle = s * invFreq[i];
                float c = (float)(Math.Cos(angle) * mscale), si = (float)(Math.Sin(angle) * mscale);
                long b = (long)s * HeadDim;
                pc[b + i] = c; pc[b + i + half] = c; ps[b + i] = si; ps[b + i + half] = si;
            }
        return (cos, sin);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
