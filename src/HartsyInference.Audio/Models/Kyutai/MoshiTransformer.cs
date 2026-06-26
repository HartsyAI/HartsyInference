using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

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
/// <c>KyutaiBackboneParityTests</c>). Generation uses <see cref="StepForward"/> (one token at a time through a
/// <c>FixedKvCache</c>); cross K/V are precomputed once via <see cref="PrecomputeCrossKv"/>. The remaining
/// host-pointer helpers are not residency hot spots: <c>SliceRows</c>/<c>FlattenAlpha</c> run once at load time
/// on host weights, and <c>SplitQkv</c>/<c>FlattenHeads</c> are single-row copies at the <c>t=1</c> step (the
/// data is already contiguous in the right layout).</para></summary>
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

    /// <summary>Per-layer cross-attention K/V, precomputed once per generation from the constant conditioning
    /// <c>cross [1,S,dim]</c> (so the projection + head-permute is NOT repeated every layer every step). Each
    /// entry is already in multi-head layout <c>[1,Heads,S,HeadDim]</c> ready for SDPA / FlashAttention.</summary>
    public sealed class CrossKvCache : IDisposable
    {
        public readonly Tensor[] K;
        public readonly Tensor[] V;
        public readonly int S;
        public CrossKvCache(int layers, int s) { K = new Tensor[layers]; V = new Tensor[layers]; S = s; }
        public void Dispose() { foreach (Tensor? t in K) t?.Dispose(); foreach (Tensor? t in V) t?.Dispose(); }
    }

    /// <summary>Projects + head-permutes the cross conditioning <paramref name="cross"/> <c>[1,S,dim]</c> into
    /// per-layer K/V <c>[1,Heads,S,HeadDim]</c> once. Cross K/V are constant across the whole generation, so this
    /// replaces the per-step re-projection that <see cref="CrossAttn"/> used to do.</summary>
    public CrossKvCache PrecomputeCrossKv(IBackend backend, Tensor cross)
    {
        int s = (int)cross.Shape[1];
        CrossKvCache c = new(_layers, s);
        for (int i = 0; i < _layers; i++)
        {
            Tensor kvFlat = WhisperOps.ProjectLinear(backend, cross, _crossKV[i]!, null, 1, s, Dim, 2 * Dim);
            (Tensor k, Tensor v) = SplitKv(kvFlat, s); kvFlat.Dispose();   // each [1,s,H,D]
            Tensor kMh = new(new TensorShape(1, Heads, s, HeadDim), DType.F32);
            Tensor vMh = new(new TensorShape(1, Heads, s, HeadDim), DType.F32);
            backend.Permute0213(kMh, k, s, Heads, HeadDim);
            backend.Permute0213(vMh, v, s, Heads, HeadDim);
            k.Dispose(); v.Dispose();
            c.K[i] = kMh; c.V[i] = vMh;
        }
        return c;
    }

    /// <summary>Runs the full backbone on a precomputed input embedding <paramref name="input"/> <c>[1,T,dim]</c>
    /// with cross-attention conditioning <paramref name="cross"/> <c>[1,S,dim]</c>. Returns the post-out_norm
    /// hidden state <c>[1,T,dim]</c>. (Full-sequence path, kept for the parity tests; generation uses
    /// <see cref="StepForward"/>.)</summary>
    public Tensor Forward(IBackend backend, Tensor input, Tensor cross)
    {
        int t = (int)input.Shape[1];
        using CrossKvCache crossKv = PrecomputeCrossKv(backend, cross);
        (Tensor cosT, Tensor sinT) = BuildRope(t);
        Tensor causal = WhisperOps.BuildCausalMask(t);

        Tensor x = new(new TensorShape(1, t, Dim), DType.F32);
        Buffer.MemoryCopy((void*)input.DataPointer, (void*)x.DataPointer, (long)t * Dim * 4, (long)t * Dim * 4);

        for (int i = 0; i < _layers; i++)
        {
            x = SelfAttn(backend, x, i, t, cosT, sinT, causal);
            x = CrossAttn(backend, x, crossKv, i, t);
            x = Gating(backend, x, i, t);
        }

        Tensor outT = new(new TensorShape(1, t, Dim), DType.F32);
        backend.RmsNorm(outT, x, _outNorm!, RmsEps);
        x.Dispose();
        cosT.Dispose(); sinT.Dispose(); causal.Dispose();
        return outT;
    }

    /// <summary>Streaming single-token decode: processes ONE new frame embedding <paramref name="frameEmbed"/>
    /// <c>[1,1,dim]</c> at absolute position <paramref name="pos"/>, appending its self-attention K/V to
    /// <paramref name="selfCache"/> and attending against the cached prefix (no causal mask needed — the buffer
    /// holds only keys <c>0..pos</c>). Cross-attention uses the precomputed <paramref name="crossKv"/>. Returns
    /// the post-out_norm hidden <c>[1,1,dim]</c>. Equivalent (to float rounding) to row <paramref name="pos"/> of
    /// <see cref="Forward"/> when fed the same prefix one token at a time. The caller must allocate
    /// <paramref name="selfCache"/> as 16 layers × <c>[1,Heads,maxSeq,HeadDim]</c> and call this with
    /// monotonically increasing <c>pos = selfCache.CurrentLength</c>.</summary>
    public Tensor StepForward(IBackend backend, Tensor frameEmbed, CrossKvCache crossKv, FixedKvCache selfCache, int pos)
    {
        (Tensor cos, Tensor sin) = BuildRopeAt(pos);

        Tensor x = new(new TensorShape(1, 1, Dim), DType.F32);
        Buffer.MemoryCopy((void*)frameEmbed.DataPointer, (void*)x.DataPointer, (long)Dim * 4, (long)Dim * 4);

        for (int i = 0; i < _layers; i++)
        {
            x = SelfAttnStep(backend, x, i, cos, sin, selfCache, pos);
            x = CrossAttn(backend, x, crossKv, i, 1);
            x = Gating(backend, x, i, 1);
        }
        selfCache.AdvanceLength(1);   // commit this token's K/V for all layers (appended at offset == pos)

        Tensor outT = new(new TensorShape(1, 1, Dim), DType.F32);
        backend.RmsNorm(outT, x, _outNorm!, RmsEps);
        x.Dispose();
        cos.Dispose(); sin.Dispose();
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

    /// <summary>Streaming self-attention for one token at position <paramref name="pos"/>: project + RoPE the
    /// single query/key, append K/V to <paramref name="cache"/> (at <c>cache.CurrentLength == pos</c>), then
    /// FlashAttention the query against the cached prefix (<c>kvLen = pos+1</c>, no causal mask: the buffer holds
    /// only past keys). Mirrors <see cref="SelfAttn"/> at <c>t=1</c>.</summary>
    private Tensor SelfAttnStep(IBackend backend, Tensor x, int layer, Tensor cos, Tensor sin, FixedKvCache cache, int pos)
    {
        Tensor pre = new(new TensorShape(1, 1, Dim), DType.F32);
        backend.RmsNorm(pre, x, _norm1[layer]!, RmsEps);
        Tensor qkv = WhisperOps.ProjectLinear(backend, pre, _selfIn[layer]!, null, 1, 1, Dim, 3 * Dim);
        pre.Dispose();
        (Tensor q, Tensor k, Tensor v) = SplitQkv(qkv, 1); qkv.Dispose();   // each [1,1,H,D]

        backend.ApplyRopeInterleaved(q, cos, sin);
        backend.ApplyRopeInterleaved(k, cos, sin);

        Tensor qMh = new(new TensorShape(1, Heads, 1, HeadDim), DType.F32);
        Tensor kMh = new(new TensorShape(1, Heads, 1, HeadDim), DType.F32);
        Tensor vMh = new(new TensorShape(1, Heads, 1, HeadDim), DType.F32);
        backend.Permute0213(qMh, q, 1, Heads, HeadDim);
        backend.Permute0213(kMh, k, 1, Heads, HeadDim);
        backend.Permute0213(vMh, v, 1, Heads, HeadDim);
        q.Dispose(); k.Dispose(); v.Dispose();

        cache.AppendStep(backend, layer, kMh, vMh);   // writes at offset == cache.CurrentLength == pos
        kMh.Dispose(); vMh.Dispose();

        Tensor attn = new(new TensorShape(1, Heads, 1, HeadDim), DType.F32);
        backend.FlashAttention(attn, qMh, cache.KeyPrefix(layer), cache.ValuePrefix(layer),
            kvLen: pos + 1, kvGroup: 1, causal: false, qOffset: 0, scale: 1f / MathF.Sqrt(HeadDim));
        qMh.Dispose();

        Tensor flat = FlattenHeads(attn, 1);
        attn.Dispose();
        Tensor o = WhisperOps.ProjectLinear(backend, flat, _selfOut[layer]!, null, 1, 1, Dim, Dim);
        flat.Dispose();
        backend.Add(o, o, x);   // residual
        x.Dispose();
        return o;
    }

    private Tensor CrossAttn(IBackend backend, Tensor x, CrossKvCache crossKv, int layer, int t)
    {
        Tensor pre = new(new TensorShape(1, t, Dim), DType.F32);
        backend.LayerNorm(pre, x, _normCrossW[layer]!, _normCrossB[layer]!, LnEps);
        Tensor qFlat = WhisperOps.ProjectLinear(backend, pre, _crossQ[layer]!, null, 1, t, Dim, Dim);
        pre.Dispose();

        // Cross-attention is RoPE-free (moshi: "rope and cross_attention makes no sense"). K/V are precomputed
        // (already in [1,Heads,S,HeadDim]); only the query is projected + permuted per call.
        Tensor q = ToHeads(qFlat, t); qFlat.Dispose();
        Tensor attn = AttendKv(backend, q, crossKv.K[layer], crossKv.V[layer], t, crossKv.S, null);
        q.Dispose();

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
        Tensor kMh = new(new TensorShape(1, Heads, tk, HeadDim), DType.F32);
        Tensor vMh = new(new TensorShape(1, Heads, tk, HeadDim), DType.F32);
        backend.Permute0213(kMh, k, tk, Heads, HeadDim);
        backend.Permute0213(vMh, v, tk, Heads, HeadDim);
        Tensor flat = AttendKv(backend, q, kMh, vMh, tq, tk, mask);
        kMh.Dispose(); vMh.Dispose();
        return flat;
    }

    /// <summary>SDPA for q <c>[1,T,H,D]</c> against already-head-permuted k/v <c>[1,H,Tk,D]</c>: permute q to
    /// <c>[1,H,T,D]</c>, attend, flatten back to <c>[1,T,dim]</c>. Used by cross-attn (K/V precomputed) and the
    /// streaming self-attn (K/V come from the cache buffer).</summary>
    private static Tensor AttendKv(IBackend backend, Tensor q, Tensor kMh, Tensor vMh, int tq, int tk, Tensor? mask)
    {
        Tensor qMh = new(new TensorShape(1, Heads, tq, HeadDim), DType.F32);
        backend.Permute0213(qMh, q, tq, Heads, HeadDim);

        Tensor attn = new(new TensorShape(1, Heads, tq, HeadDim), DType.F32);
        backend.ScaledDotProductAttention(attn, qMh, kMh, vMh, mask, 1f / MathF.Sqrt(HeadDim));
        qMh.Dispose();

        Tensor flat = FlattenHeads(attn, tq);
        attn.Dispose();
        return flat;
    }

    /// <summary>Flattens an attention output <c>[1,Heads,tq,HeadDim]</c> back to <c>[1,tq,dim]</c>. At
    /// <c>tq==1</c> this is a single contiguous copy (the per-head rows are already adjacent).</summary>
    private static Tensor FlattenHeads(Tensor attn, int tq)
    {
        Tensor flat = new(new TensorShape(1, tq, Dim), DType.F32);
        float* ap = (float*)attn.DataPointer; float* fp = (float*)flat.DataPointer;
        for (int h = 0; h < Heads; h++)
            for (int tt = 0; tt < tq; tt++)
                Buffer.MemoryCopy(ap + (((long)h * tq + tt) * HeadDim), fp + (((long)tt * Heads + h) * HeadDim), HeadDim * 4, HeadDim * 4);
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

    /// <summary>Single-position RoPE: the one cos/sin row <c>[1,1,HeadDim]</c> that <see cref="BuildRope"/> would
    /// produce at row <paramref name="pos"/>. Bit-identical to that row (same <c>angle = pos·invFreq[i]</c> and
    /// the same duplicate-into-<c>[i]</c>/<c>[i+half]</c> layout), so the streaming path matches the full path.</summary>
    private static (Tensor cos, Tensor sin) BuildRopeAt(int pos)
    {
        (double[] invFreq, double mscale) = RopeFrequencyBuilder.Build(HeadDim, RopeTheta, null, pos + 1);
        int half = HeadDim / 2;
        Tensor cos = new(new TensorShape(1, 1, HeadDim), DType.F32);
        Tensor sin = new(new TensorShape(1, 1, HeadDim), DType.F32);
        float* pc = (float*)cos.DataPointer; float* ps = (float*)sin.DataPointer;
        for (int i = 0; i < half; i++)
        {
            double angle = pos * invFreq[i];
            float c = (float)(Math.Cos(angle) * mscale), si = (float)(Math.Sin(angle) * mscale);
            pc[i] = c; pc[i + half] = c; ps[i] = si; ps[i + half] = si;
        }
        return (cos, sin);
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
