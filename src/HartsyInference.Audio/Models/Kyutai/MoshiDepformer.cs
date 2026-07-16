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
    private readonly Tensor?[,] _selfIn = new Tensor?[Layers, Sets];   // per-set QKV proj [3072,1024]
    private readonly Tensor?[,] _selfOut = new Tensor?[Layers, Sets];  // per-set out proj [1024,1024]
    private readonly Tensor?[,] _gateInGate = new Tensor?[Layers, Sets];  // gate half of gating.linear_in [2048,1024]
    private readonly Tensor?[,] _gateInUp = new Tensor?[Layers, Sets];    // up half [2048,1024]
    private readonly Tensor?[,] _gateOut = new Tensor?[Layers, Sets];     // [1024,2048]
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
            // Pre-slice the per-weight-set projections at load. Slicing them fresh each Block call allocated a new
            // host tensor per (codebook,layer) every frame, so the device weight cache (keyed by tensor identity)
            // NEVER hit and re-uploaded the weight on every op — the dominant generation cost (H2D_MISS_BIG).
            Tensor inAll = WhisperOps.EnsureF32(w[$"{p}.self_attn.in_proj_weight"]);   // [11·3·Dim, Dim]
            Tensor outAll = WhisperOps.EnsureF32(w[$"{p}.self_attn.out_proj.weight"]); // [11·Dim, Dim]
            _norm1[l] = Flatten(WhisperOps.EnsureF32(w[$"{p}.norm1.alpha"]));
            _norm2[l] = Flatten(WhisperOps.EnsureF32(w[$"{p}.norm2.alpha"]));
            for (int s = 0; s < Sets; s++)
            {
                _selfIn[l, s] = SliceRows(inAll, s * 3 * Dim, 3 * Dim, Dim);
                _selfOut[l, s] = SliceRows(outAll, s * Dim, Dim, Dim);
                // Pre-split the fused gate|up projection ([2·GateInner, Dim]) so the SwiGLU runs entirely on the
                // device (Silu + Mul) instead of a per-call host loop that drains the GPU for every codebook.
                Tensor gin = WhisperOps.EnsureF32(w[$"{p}.gating.{s}.linear_in.weight"]);
                _gateInGate[l, s] = SliceRows(gin, 0, GateInner, Dim);
                _gateInUp[l, s] = SliceRows(gin, GateInner, GateInner, Dim);
                _gateOut[l, s] = WhisperOps.EnsureF32(w[$"{p}.gating.{s}.linear_out.weight"]);
            }
        }
        for (int cb = 0; cb < DepQ; cb++) _heads[cb] = WhisperOps.EnsureF32(w[$"linears.{cb}.weight"]);
    }

    /// <summary>Greedy (argmax) decode of all 32 codebooks for one temporal frame; returns the per-codebook
    /// logits <c>[DepQ, Card]</c>. <paramref name="textToken"/> seeds the cb-0 step. Deterministic, for parity.</summary>
    public Tensor DecodeFrameGreedy(IBackend backend, Tensor transformerOut, int textToken, out int[] tokens,
        float temp = 0f, int topK = 0, Random? rng = null, int[]? forcedPrev = null)
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
                ReadOnlySpan<float> lgSpan = new((void*)lg.DataPointer, Card);
                int tok = temp > 0f && rng is not null ? SampleTopK(lgSpan, temp, topK, rng) : ArgMax(lgSpan);
                lg.Dispose();
                tokens[cb] = tok;
                // Teacher forcing (parity): feed the reference's previous-codebook token as the next step's input,
                // so each codebook's logits are compared under identical context (a sampling model never follows
                // the greedy cascade, so greedy-cascade divergence is not itself a bug).
                prev = forcedPrev is not null ? forcedPrev[cb] : tok;
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

        // Packed QKV via the pre-sliced per-set in_proj [3·dim, dim] (stable → device-cacheable).
        Tensor qkv = WhisperOps.ProjectLinear(backend, pre, _selfIn[layer, set]!, null, 1, 1, Dim, 3 * Dim);
        pre.Dispose();

        Tensor q = HeadSlice(qkv, 0), k = HeadSlice(qkv, Dim), v = HeadSlice(qkv, 2 * Dim);
        qkv.Dispose();
        WriteStep(kCache, k, cb); k.Dispose();
        WriteStep(vCache, v, cb); v.Dispose();
        Tensor kPre = Prefix(kCache, cb + 1), vPre = Prefix(vCache, cb + 1);

        Tensor attn = new(new TensorShape(1, Heads, 1, HeadDim), DType.F32);
        backend.ScaledDotProductAttention(attn, q, kPre, vPre, null, 1f / MathF.Sqrt(HeadDim));
        q.Dispose(); kPre.Dispose(); vPre.Dispose();
        Tensor attnFlat = HeadsToFlat(attn); attn.Dispose();

        Tensor o = WhisperOps.ProjectLinear(backend, attnFlat, _selfOut[layer, set]!, null, 1, 1, Dim, Dim);
        attnFlat.Dispose();
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
        // SwiGLU: silu(gate_proj(x)) * up_proj(x), all on-device (no host drain).
        TensorShape sh = new(1, 1, GateInner);
        Tensor gate = WhisperOps.ProjectLinear(backend, x, _gateInGate[layer, set]!, null, 1, 1, Dim, GateInner);
        Tensor up = WhisperOps.ProjectLinear(backend, x, _gateInUp[layer, set]!, null, 1, 1, Dim, GateInner);
        Tensor act = new(sh, DType.F32);
        backend.Silu(act, gate); gate.Dispose();
        backend.Mul(act, act, up); up.Dispose();
        Tensor o = WhisperOps.ProjectLinear(backend, act, _gateOut[layer, set]!, null, 1, 1, GateInner, Dim);
        act.Dispose();
        return o;
    }

    // Demuxing text embedding (depformer cb-0 input): the fed text token packs two streams as
    // (second+1)·card + main; y = out1(emb[main]) + (second≥0 ? out2(emb[second]) : 0). Mirrors the temporal
    // text_emb demux. (The depformer parity test only exercised a small non-multiplexed token, so this path
    // was previously untested.)
    private Tensor TextEmbed(IBackend backend, int token)
    {
        if (token < 0) return new Tensor(new TensorShape(1, 1, Dim), DType.F32);   // zero token → zeros
        int card = TextCard + 1;
        int main = token % card, second = token / card - 1;
        Tensor lookMain = LowRankLookup(_textW!, main, LowRank);   // [1,1,128]
        Tensor y = WhisperOps.ProjectLinear(backend, lookMain, _textOut1!, null, 1, 1, LowRank, Dim);
        lookMain.Dispose();
        if (second >= 0)
        {
            Tensor lookSecond = LowRankLookup(_textW!, second, LowRank);
            Tensor r = WhisperOps.ProjectLinear(backend, lookSecond, _textOut2!, null, 1, 1, LowRank, Dim);
            lookSecond.Dispose();
            backend.Add(y, y, r); r.Dispose();
        }
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

    /// <summary>Top-k temperature sampling (moshi <c>sample_token</c>): scale logits by <paramref name="temp"/>,
    /// keep the <paramref name="topK"/> highest, softmax over them, then multinomial-sample with
    /// <paramref name="rng"/>. Kyutai TTS was trained for sampling (audio temp 0.8 / top-k 250); greedy argmax
    /// collapses the code cascade to non-speech.</summary>
    private static int SampleTopK(ReadOnlySpan<float> logits, float temp, int topK, Random rng)
    {
        int n = logits.Length;
        int k = topK <= 0 ? n : Math.Min(topK, n);
        // Indices of the k largest logits (k is small vs n; a full index sort is fine here).
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        float[] vals = new float[n];
        for (int i = 0; i < n; i++) vals[i] = logits[i];
        Array.Sort(idx, (a, b) => vals[b].CompareTo(vals[a]));   // descending by logit

        float max = vals[idx[0]] / temp;
        double sum = 0;
        double[] p = new double[k];
        for (int j = 0; j < k; j++) { p[j] = Math.Exp(vals[idx[j]] / temp - max); sum += p[j]; }
        double r = rng.NextDouble() * sum, acc = 0;
        for (int j = 0; j < k; j++) { acc += p[j]; if (r <= acc) return idx[j]; }
        return idx[k - 1];
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
