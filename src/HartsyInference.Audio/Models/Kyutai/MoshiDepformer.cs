using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Kyutai;

/// <summary>Moshi/Kyutai depth transformer ("depformer") with the REAL checkpoint layout (the older
/// <see cref="MoshiDepthTransformer"/> assumed a Qwen2-style per-(set,layer) layout that the checkpoint does not
/// use). Given one temporal-frame context <c>[1,1,2048]</c> it autoregressively predicts the 32 Mimi codebooks.
///
/// <para>Per codebook <c>cb</c> (weight-set <c>s = schedule[cb]</c>): project the temporal context through the
/// per-set <c>depformer_in[s]</c> (2048→1024); add the previous token's low-rank embedding (the text demux
/// embedding for cb 0, else <c>depformer_emb[cb-1]</c>); run the 4-layer depth transformer (RoPE-free, causal
/// over depth steps 0..cb, each layer slicing its PACKED <c>self_attn.in_proj_weight</c> [11·3072,1024] /
/// <c>out_proj.weight</c> [11·1024,1024] by <c>s</c> and using the per-set <c>gating.{s}</c>); project through
/// the per-codebook head <c>linears[cb]</c> (1024→2048). Norms are RMSNorm-alpha (eps 1e-5); the
/// <c>depformer_norms</c> before the head are Identity. Validated against the real checkpoint in
/// <c>KyutaiDepformerParityTests</c>. TODO(gpu-residency): the head-split / cache helpers loop on host pointers.</para></summary>
public sealed unsafe class MoshiDepformer : IDisposable
{
    public const int Dim = 1024, Heads = 16, HeadDim = 64, Layers = 4, Sets = 11, DepQ = 32;
    public const int LowRank = 128, MainDim = 2048, Card = 2048, GateInner = 2048, TextCard = 8000;
    private const float RmsEps = 1e-5f;
    private static readonly int[] Schedule = BuildSchedule();

    private readonly Tensor?[] _inProj = new Tensor?[Sets];        // depformer_in[s] [1024,2048]
    private readonly Tensor?[] _embW = new Tensor?[DepQ - 1];      // depformer_emb[k] lookup [2049,128]
    private readonly Tensor?[] _embLr = new Tensor?[DepQ - 1];     // depformer_emb[k].low_rank [1024,128]
    private Tensor? _textW, _textOut1, _textOut2;                  // depformer_text_emb (demux)
    private readonly Tensor?[] _selfIn = new Tensor?[Layers];      // [11·3072,1024]
    private readonly Tensor?[] _selfOut = new Tensor?[Layers];     // [11·1024,1024]
    private readonly Tensor?[,] _gateIn = new Tensor?[Layers, Sets];   // [4096,1024]
    private readonly Tensor?[,] _gateOut = new Tensor?[Layers, Sets];  // [1024,2048]
    private readonly Tensor?[] _norm1 = new Tensor?[Layers], _norm2 = new Tensor?[Layers];
    private readonly Tensor?[] _heads = new Tensor?[DepQ];         // linears[cb] [2048,1024]
    private int _disposed;

    private static int[] BuildSchedule()
    {
        int[] s = new int[DepQ];
        for (int k = 0; k < DepQ; k++) s[k] = k < 8 ? k : 8 + (k - 8) / 8;
        return s;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        for (int s = 0; s < Sets; s++) _inProj[s] = WhisperOps.EnsureF32(w[$"depformer_in.{s}.weight"]);
        for (int k = 0; k < DepQ - 1; k++)
        {
            _embW[k] = WhisperOps.EnsureF32(w[$"depformer_emb.{k}.weight"]);
            _embLr[k] = WhisperOps.EnsureF32(w[$"depformer_emb.{k}.low_rank.weight"]);
        }
        _textW = WhisperOps.EnsureF32(w["depformer_text_emb.weight"]);
        _textOut1 = WhisperOps.EnsureF32(w["depformer_text_emb.out1.weight"]);
        _textOut2 = WhisperOps.EnsureF32(w["depformer_text_emb.out2.weight"]);
        for (int l = 0; l < Layers; l++)
        {
            string p = $"depformer.layers.{l}";
            _selfIn[l] = WhisperOps.EnsureF32(w[$"{p}.self_attn.in_proj_weight"]);
            _selfOut[l] = WhisperOps.EnsureF32(w[$"{p}.self_attn.out_proj.weight"]);
            _norm1[l] = Flatten(WhisperOps.EnsureF32(w[$"{p}.norm1.alpha"]));
            _norm2[l] = Flatten(WhisperOps.EnsureF32(w[$"{p}.norm2.alpha"]));
            for (int s = 0; s < Sets; s++)
            {
                _gateIn[l, s] = WhisperOps.EnsureF32(w[$"{p}.gating.{s}.linear_in.weight"]);
                _gateOut[l, s] = WhisperOps.EnsureF32(w[$"{p}.gating.{s}.linear_out.weight"]);
            }
        }
        for (int cb = 0; cb < DepQ; cb++) _heads[cb] = WhisperOps.EnsureF32(w[$"linears.{cb}.weight"]);
    }

    /// <summary>Greedy (argmax) decode of all 32 codebooks for one temporal frame; returns the per-codebook
    /// logits <c>[DepQ, Card]</c>. <paramref name="textToken"/> seeds the cb-0 step. Deterministic, for parity.</summary>
    public Tensor DecodeFrameGreedy(IBackend backend, Tensor transformerOut, int textToken, out int[] tokens)
    {
        Tensor logits = new(new TensorShape(DepQ, Card), DType.F32);
        tokens = new int[DepQ];
        Tensor[] kCache = new Tensor[Layers], vCache = new Tensor[Layers];
        for (int l = 0; l < Layers; l++)
        {
            kCache[l] = new Tensor(new TensorShape(1, Heads, DepQ, HeadDim), DType.F32);
            vCache[l] = new Tensor(new TensorShape(1, Heads, DepQ, HeadDim), DType.F32);
        }
        try
        {
            int prev = textToken;
            for (int cb = 0; cb < DepQ; cb++)
            {
                int set = Schedule[cb];
                Tensor depIn = WhisperOps.ProjectLinear(backend, transformerOut, _inProj[set]!, null, 1, 1, MainDim, Dim);
                Tensor emb = cb == 0 ? TextEmbed(backend, prev) : CodeEmbed(backend, cb - 1, prev);
                Tensor x = new(new TensorShape(1, 1, Dim), DType.F32);
                backend.Add(x, depIn, emb); depIn.Dispose(); emb.Dispose();

                for (int l = 0; l < Layers; l++)
                    x = Block(backend, x, l, set, cb, kCache[l], vCache[l]);

                Tensor lg = WhisperOps.ProjectLinear(backend, x, _heads[cb]!, null, 1, 1, Dim, Card);
                x.Dispose();
                Buffer.MemoryCopy((void*)lg.DataPointer, (float*)logits.DataPointer + (long)cb * Card, Card * 4, Card * 4);
                int tok = ArgMax(new ReadOnlySpan<float>((void*)lg.DataPointer, Card));
                lg.Dispose();
                tokens[cb] = tok; prev = tok;
            }
            return logits;
        }
        finally
        {
            for (int l = 0; l < Layers; l++) { kCache[l].Dispose(); vCache[l].Dispose(); }
        }
    }

    private Tensor Block(IBackend backend, Tensor x, int layer, int set, int cb, Tensor kCache, Tensor vCache)
    {
        TensorShape sh = new(1, 1, Dim);
        Tensor pre = new(sh, DType.F32);
        backend.RmsNorm(pre, x, _norm1[layer]!, RmsEps);

        // Packed QKV: slice the set's [3·dim, dim] rows from in_proj [11·3·dim, dim].
        Tensor inW = SliceRows(_selfIn[layer]!, set * 3 * Dim, 3 * Dim, Dim);
        Tensor qkv = WhisperOps.ProjectLinear(backend, pre, inW, null, 1, 1, Dim, 3 * Dim);
        inW.Dispose(); pre.Dispose();

        Tensor q = HeadSlice(qkv, 0), k = HeadSlice(qkv, Dim), v = HeadSlice(qkv, 2 * Dim);
        qkv.Dispose();
        WriteStep(kCache, k, cb); k.Dispose();
        WriteStep(vCache, v, cb); v.Dispose();
        Tensor kPre = Prefix(kCache, cb + 1), vPre = Prefix(vCache, cb + 1);

        Tensor attn = new(new TensorShape(1, Heads, 1, HeadDim), DType.F32);
        backend.ScaledDotProductAttention(attn, q, kPre, vPre, null, 1f / MathF.Sqrt(HeadDim));
        q.Dispose(); kPre.Dispose(); vPre.Dispose();
        Tensor attnFlat = HeadsToFlat(attn); attn.Dispose();

        Tensor outW = SliceRows(_selfOut[layer]!, set * Dim, Dim, Dim);
        Tensor o = WhisperOps.ProjectLinear(backend, attnFlat, outW, null, 1, 1, Dim, Dim);
        outW.Dispose(); attnFlat.Dispose();
        Tensor afterAttn = new(sh, DType.F32);
        backend.Add(afterAttn, x, o); o.Dispose(); x.Dispose();

        Tensor pre2 = new(sh, DType.F32);
        backend.RmsNorm(pre2, afterAttn, _norm2[layer]!, RmsEps);
        Tensor mlp = Gating(backend, pre2, layer, set); pre2.Dispose();
        Tensor result = new(sh, DType.F32);
        backend.Add(result, afterAttn, mlp); afterAttn.Dispose(); mlp.Dispose();
        return result;
    }

    private Tensor Gating(IBackend backend, Tensor x, int layer, int set)
    {
        Tensor gu = WhisperOps.ProjectLinear(backend, x, _gateIn[layer, set]!, null, 1, 1, Dim, 2 * GateInner);
        Tensor act = new(new TensorShape(1, 1, GateInner), DType.F32);
        float* g = (float*)gu.DataPointer; float* a = (float*)act.DataPointer;
        for (int c = 0; c < GateInner; c++) { float gate = g[c]; a[c] = (gate / (1f + MathF.Exp(-gate))) * g[GateInner + c]; }
        gu.Dispose();
        Tensor o = WhisperOps.ProjectLinear(backend, act, _gateOut[layer, set]!, null, 1, 1, GateInner, Dim);
        act.Dispose();
        return o;
    }

    // text demux embedding: feeding a normal token (< TextCard) gives out1(emb_lookup(token)).
    private Tensor TextEmbed(IBackend backend, int token)
    {
        Tensor look = LowRankLookup(_textW!, token, LowRank);   // [1,1,128]
        Tensor y = WhisperOps.ProjectLinear(backend, look, _textOut1!, null, 1, 1, LowRank, Dim);
        look.Dispose();
        return y;
    }

    private Tensor CodeEmbed(IBackend backend, int idx, int token)
    {
        Tensor look = LowRankLookup(_embW[idx]!, token, LowRank);   // [1,1,128]
        Tensor y = WhisperOps.ProjectLinear(backend, look, _embLr[idx]!, null, 1, 1, LowRank, Dim);
        look.Dispose();
        return y;
    }

    private static Tensor LowRankLookup(Tensor table, int token, int lr)
    {
        Tensor outT = new(new TensorShape(1, 1, lr), DType.F32);
        Buffer.MemoryCopy((float*)table.DataPointer + (long)token * lr, (void*)outT.DataPointer, lr * 4, lr * 4);
        return outT;
    }

    // [1,1,3·dim] column slice → [1,Heads,1,HeadDim].
    private static Tensor HeadSlice(Tensor qkv, int colOff)
    {
        Tensor outT = new(new TensorShape(1, Heads, 1, HeadDim), DType.F32);
        Buffer.MemoryCopy((float*)qkv.DataPointer + colOff, (void*)outT.DataPointer, Dim * 4, Dim * 4);
        return outT;
    }

    private static Tensor HeadsToFlat(Tensor attn)
    {
        Tensor outT = new(new TensorShape(1, 1, Dim), DType.F32);
        Buffer.MemoryCopy((void*)attn.DataPointer, (void*)outT.DataPointer, Dim * 4, Dim * 4);
        return outT;
    }

    private static void WriteStep(Tensor cache, Tensor proj, int pos)
    {
        float* cp = (float*)cache.DataPointer; float* pp = (float*)proj.DataPointer;
        for (int h = 0; h < Heads; h++)
            Buffer.MemoryCopy(pp + (long)h * HeadDim, cp + ((long)h * DepQ + pos) * HeadDim, HeadDim * 4, HeadDim * 4);
    }

    private static Tensor Prefix(Tensor cache, int len)
    {
        Tensor outT = new(new TensorShape(1, Heads, len, HeadDim), DType.F32);
        float* cp = (float*)cache.DataPointer; float* op = (float*)outT.DataPointer;
        for (int h = 0; h < Heads; h++)
            Buffer.MemoryCopy(cp + (long)h * DepQ * HeadDim, op + (long)h * len * HeadDim, (long)len * HeadDim * 4, (long)len * HeadDim * 4);
        return outT;
    }

    private static Tensor SliceRows(Tensor w, int r0, int rows, int inDim)
    {
        Tensor outT = new(new TensorShape(rows, inDim), DType.F32);
        Buffer.MemoryCopy((float*)w.DataPointer + (long)r0 * inDim, (void*)outT.DataPointer, (long)rows * inDim * 4, (long)rows * inDim * 4);
        return outT;
    }

    private static Tensor Flatten(Tensor alpha)
    {
        Tensor outT = new(new TensorShape(Dim), DType.F32);
        Buffer.MemoryCopy((void*)alpha.DataPointer, (void*)outT.DataPointer, (long)Dim * 4, (long)Dim * 4);
        return outT;
    }

    private static int ArgMax(ReadOnlySpan<float> v)
    {
        int best = 0; float bv = v[0];
        for (int i = 1; i < v.Length; i++) if (v[i] > bv) { bv = v[i]; best = i; }
        return best;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
