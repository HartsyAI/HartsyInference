using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Embeddings;

/// <summary>A BERT-family text-embedding model loaded from a llama.cpp encoder GGUF (<c>bert</c>, <c>nomic-bert</c>, …); runs the bidirectional post-norm encoder, pools (CLS/mean per <c>pooling_type</c>), then L2-normalizes.</summary>
/// <remarks>Config-driven over family differences detected from metadata + tensor presence:
/// <list type="bullet">
/// <item>position: absolute table (bge, all-MiniLM) vs rotary (nomic-bert)</item>
/// <item>QKV: separate <c>attn_q/k/v</c> (bge) vs fused <c>attn_qkv</c> (nomic, split at load)</item>
/// <item>MLP: GELU (bge) vs GeGLU <c>ffn_gate·ffn_up</c> (nomic)</item>
/// <item>biases: present (bge) vs none on the linears (nomic)</item>
/// </list>
/// Reuses the engine's Linear / LayerNorm / FlashAttention (non-causal) primitives.</remarks>
public sealed unsafe class BertEmbeddingModel : IDisposable
{
    private readonly GgufModelLoader.LoadedGgufModel _handle;
    private readonly IReadOnlyDictionary<string, Tensor> _w;
    private int _disposed;

    public int Hidden { get; }
    public int NumLayers { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }
    public int Intermediate { get; }
    public int MaxPos { get; }
    public float Eps { get; }
    /// <summary>0 = none, 1 = mean, 2 = CLS (llama.cpp pooling type).</summary>
    public int PoolingType { get; }
    public bool UseRope { get; }       // nomic-bert: rotary position (no absolute table)
    public float RopeBase { get; }
    public bool UseGeGlu { get; }      // gated MLP (nomic = SwiGLU/SiLU gate; jina-bert-v2 = GEGLU/GELU gate)
    public bool GateIsGelu { get; }    // jina-bert-v2: the gate activation is GELU (GEGLU), not SiLU (SwiGLU)
    public bool UseAlibi { get; }      // jina-bert-v2: symmetric ALiBi bias, no position table, no RoPE
    private readonly Tensor? _alibiSlopes;
    public bool PreNormRms { get; }    // neo-bert: pre-norm RMSNorm blocks + a single final RMSNorm (no per-sublayer post-LN, no token-type/embedding norm)
    public int MoeEveryN { get; }      // nomic-bert-moe: every Nth block replaces the dense FFN with a routed MoE FFN (0 = dense everywhere)
    public int NumExperts { get; }
    public int ExpertsUsed { get; }
    private bool IsMoeLayer(int i) => MoeEveryN > 0 && (i % MoeEveryN) == (MoeEveryN - 1);
    public string[]? Vocab { get; }
    /// <summary>True if this GGUF carries a reranker classification head (<c>cls.*</c>) — call <see cref="Score"/>.</summary>
    public bool HasRerankHead { get; }

    private BertEmbeddingModel(GgufModelLoader.LoadedGgufModel handle, IReadOnlyDictionary<string, Tensor> w,
        int hidden, int layers, int heads, int inter, int maxPos, float eps, int pooling, string[]? vocab,
        bool useRope, float ropeBase, bool useGeGlu, bool hasRerankHead, bool gateIsGelu, bool useAlibi, Tensor? alibiSlopes,
        bool preNormRms, int moeEveryN, int numExperts, int expertsUsed)
    {
        _handle = handle; _w = w;
        Hidden = hidden; NumLayers = layers; NumHeads = heads; HeadDim = hidden / heads; Intermediate = inter;
        MaxPos = maxPos; Eps = eps <= 0f ? 1e-12f : eps; PoolingType = pooling; Vocab = vocab;
        UseRope = useRope; RopeBase = ropeBase; UseGeGlu = useGeGlu; HasRerankHead = hasRerankHead;
        GateIsGelu = gateIsGelu; UseAlibi = useAlibi; _alibiSlopes = alibiSlopes; PreNormRms = preNormRms;
        MoeEveryN = moeEveryN; NumExperts = numExperts; ExpertsUsed = expertsUsed;
    }

    /// <summary>Copies <paramref name="rows"/> row-major <c>[·, inDim]</c> rows from <paramref name="src"/>/<paramref name="srcR0"/> to <paramref name="dst"/>/<paramref name="dstR0"/>; used to de-interleave neo-bert's per-head fused QKV into contiguous q/k/v.</summary>
    private static void CopyRows(Tensor src, int srcR0, Tensor dst, int dstR0, int rows, int inDim)
    {
        Buffer.MemoryCopy((byte*)src.DataPointer + (long)srcR0 * inDim * 4, (byte*)dst.DataPointer + (long)dstR0 * inDim * 4,
            (long)rows * inDim * 4, (long)rows * inDim * 4);
    }

    /// <summary>Owned copy of <paramref name="count"/> elements of a 1-D tensor starting at <paramref name="start"/>.</summary>
    private static Tensor SliceVec1d(Tensor t, int start, int count)
    {
        Tensor outp = new(new TensorShape(count), DType.F32);
        Buffer.MemoryCopy((byte*)t.DataPointer + (long)start * 4, (void*)outp.DataPointer, (long)count * 4, (long)count * 4);
        return outp;
    }

    /// <summary>Slices rows <c>[r0, r0+rows)</c> of a row-major <c>[R, in]</c> weight into an owned <c>[rows, in]</c>.</summary>
    private static Tensor SliceRows(Tensor t, int r0, int rows)
    {
        int inDim = (int)t.Shape[1];
        Tensor outp = new(new TensorShape(rows, inDim), DType.F32);
        Buffer.MemoryCopy((byte*)t.DataPointer + (long)r0 * inDim * 4, (void*)outp.DataPointer, (long)rows * inDim * 4, (long)rows * inDim * 4);
        return outp;
    }

    public static BertEmbeddingModel Load(string ggufPath)
    {
        (Dictionary<string, Tensor> w, GgufModelLoader.LoadedGgufModel handle) = GgufModelLoader.LoadDequantized(ggufPath, DType.F32);
        try
        {
            string arch = (handle.Metadata.GetString("general.architecture") ?? "bert").ToLowerInvariant();
            // nn.Linear weights are ggml ne=[in,out] with raw bytes already row-major [out,in] → relabel for Linear.
            foreach (string key in w.Keys.ToList())
            {
                if (!key.EndsWith(".weight", StringComparison.Ordinal) || w[key].Shape.Rank != 2) continue;
                if (key.Contains("attn_q") || key.Contains("attn_k") || key.Contains("attn_v") || key.Contains("attn_qkv")
                    || key.Contains("attn_output") || key.Contains("ffn_up") || key.Contains("ffn_down") || key.Contains("ffn_gate")
                    || key == "cls.weight" || key == "cls.output.weight")   // reranker classification head
                    w[key] = TensorCasts.RelabelRank2Copy(w[key]);
            }
            GgufMetadata m = handle.Metadata;
            int hidden = (int)m.GetUInt32($"{arch}.embedding_length");
            int layers = (int)m.GetUInt32($"{arch}.block_count");
            int heads = (int)m.GetUInt32($"{arch}.attention.head_count");
            int inter = (int)m.GetUInt32($"{arch}.feed_forward_length");
            int maxPos = (int)m.GetUInt32($"{arch}.context_length", 512u);
            // neo-bert: pre-norm RMSNorm encoder (RoPE, fused QKV, SwiGLU, final-norm only) — uses rms_epsilon.
            bool isNeoBert = arch == "neo-bert";
            float eps = isNeoBert
                ? m.GetFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-6f)
                : m.GetFloat32($"{arch}.attention.layer_norm_epsilon", 1e-12f);
            int pooling = (int)m.GetUInt32($"{arch}.pooling_type", 2u);
            float ropeBase = m.GetFloat32($"{arch}.rope.freq_base", 0f);
            bool useRope = !w.ContainsKey("position_embd.weight") && ropeBase > 0f;
            bool useGeGlu = w.ContainsKey("blk.0.ffn_gate.weight");
            // jina-bert-v2: no position table, no RoPE — bidirectional symmetric ALiBi instead, and its gated MLP is
            // GEGLU (GELU gate), not nomic-bert's SwiGLU (SiLU gate). ALiBi max bias is hardcoded 8 in llama.cpp.
            bool isJina = arch == "jina-bert-v2";
            bool useAlibi = isJina;
            bool gateIsGelu = isJina;
            // neo-bert: SwiGLU stored as a single fused ffn_up of width 2·ffn (gate | up). Split it into the
            // gate/up halves the gated MLP path expects (SiLU gate). Mirrors the fused-QKV split below.
            if (isNeoBert)
            {
                for (int i = 0; i < layers; i++)
                {
                    string upKey = $"blk.{i}.ffn_up.weight";
                    if (!w.ContainsKey(upKey)) continue;
                    Tensor fu = w[upKey];   // already relabeled to [2·ffn, hidden]
                    ((Dictionary<string, Tensor>)w)[$"blk.{i}.ffn_gate.weight"] = SliceRows(fu, 0, inter);
                    ((Dictionary<string, Tensor>)w)[upKey] = SliceRows(fu, inter, inter);
                }
                useGeGlu = true;   // gated (SiLU) MLP after the split
            }
            Tensor? alibiSlopes = null;
            if (useAlibi)
            {
                float[] slopes = TransformerConfig.ComputeAlibiSlopes(heads, m.GetFloat32($"{arch}.attention.max_alibi_bias", 8f));
                alibiSlopes = new Tensor(new TensorShape(heads), DType.F32);
                fixed (float* sp = slopes) Buffer.MemoryCopy(sp, (void*)alibiSlopes.DataPointer, (long)heads * 4, (long)heads * 4);
            }

            // Fused QKV → separate q/k/v. Two different output layouts:
            //   nomic  → [all_q | all_k | all_v] contiguous (order q,k,v).
            //   neo-bert → PER-HEAD interleaved [h0:(q|k|v) | h1:(q|k|v) | …] (its qkv reshapes to
            //              [heads, 3·dim_head] then chunks 3) — de-interleave by gathering each head's q/k/v slice.
            int dh = hidden / heads;
            for (int i = 0; i < layers; i++)
            {
                string qkv = $"blk.{i}.attn_qkv.weight";
                if (!w.ContainsKey(qkv)) continue;
                Tensor f = w[qkv];
                if (isNeoBert)
                {
                    Tensor q = new(new TensorShape(hidden, hidden), DType.F32);
                    Tensor k = new(new TensorShape(hidden, hidden), DType.F32);
                    Tensor v = new(new TensorShape(hidden, hidden), DType.F32);
                    for (int hh = 0; hh < heads; hh++)
                    {
                        CopyRows(f, hh * 3 * dh + 0 * dh, q, hh * dh, dh, hidden);
                        CopyRows(f, hh * 3 * dh + 1 * dh, k, hh * dh, dh, hidden);
                        CopyRows(f, hh * 3 * dh + 2 * dh, v, hh * dh, dh, hidden);
                    }
                    ((Dictionary<string, Tensor>)w)[$"blk.{i}.attn_q.weight"] = q;
                    ((Dictionary<string, Tensor>)w)[$"blk.{i}.attn_k.weight"] = k;
                    ((Dictionary<string, Tensor>)w)[$"blk.{i}.attn_v.weight"] = v;
                }
                else
                {
                    ((Dictionary<string, Tensor>)w)[$"blk.{i}.attn_q.weight"] = SliceRows(f, 0, hidden);
                    ((Dictionary<string, Tensor>)w)[$"blk.{i}.attn_k.weight"] = SliceRows(f, hidden, hidden);
                    ((Dictionary<string, Tensor>)w)[$"blk.{i}.attn_v.weight"] = SliceRows(f, 2 * hidden, hidden);
                    string qkvB = $"blk.{i}.attn_qkv.bias";   // nomic-bert-moe: fused QKV bias → split alongside (order q,k,v)
                    if (w.ContainsKey(qkvB))
                    {
                        Tensor fb = w[qkvB];
                        ((Dictionary<string, Tensor>)w)[$"blk.{i}.attn_q.bias"] = SliceVec1d(fb, 0, hidden);
                        ((Dictionary<string, Tensor>)w)[$"blk.{i}.attn_k.bias"] = SliceVec1d(fb, hidden, hidden);
                        ((Dictionary<string, Tensor>)w)[$"blk.{i}.attn_v.bias"] = SliceVec1d(fb, 2 * hidden, hidden);
                    }
                }
            }

            // Reranker out_proj (hidden→1) is stored as a rank-1 [hidden] vector → reshape to [1, hidden] for Linear.
            if (w.ContainsKey("cls.output.weight") && w["cls.output.weight"].Shape.Rank == 1)
            {
                Tensor o = w["cls.output.weight"];
                Tensor r = new(new TensorShape(1, (int)o.Shape[0]), DType.F32);
                Buffer.MemoryCopy((void*)o.DataPointer, (void*)r.DataPointer, o.ElementCount * 4, o.ElementCount * 4);
                ((Dictionary<string, Tensor>)w)["cls.output.weight"] = r;
            }
            // nomic-bert-moe: every Nth block is a routed MoE FFN (router ffn_gate_inp → top-k of expert_count
            // non-gated GELU experts). The stacked expert tensors are expert-major ([E, out, in] in memory), so each
            // expert's [out, in] Linear weight is a contiguous byte-range; the router weight relabels [in,E]→[E,in].
            int moeEveryN = (int)m.GetUInt32($"{arch}.moe_every_n_layers", 0u);
            int numExperts = (int)m.GetUInt32($"{arch}.expert_count", 0u);
            int expertsUsed = (int)m.GetUInt32($"{arch}.expert_used_count", 2u);
            if (moeEveryN > 0 && numExperts > 0)
            {
                // Note: ffn_gate_inp (rank-2 router) is already relabeled [in,E]→[E,in] by the relabel loop above;
                // the rank-3 *_exps tensors are skipped there (Rank != 2) and split per-expert here.
                for (int i = 0; i < layers; i++)
                {
                    string upx = $"blk.{i}.ffn_up_exps.weight", dnx = $"blk.{i}.ffn_down_exps.weight";
                    if (!w.ContainsKey(upx)) continue;
                    Tensor up = w[upx].Reshape(new TensorShape(numExperts * inter, hidden));    // [E·ff, hidden] = per-expert [ff,hidden]
                    Tensor dn = w[dnx].Reshape(new TensorShape(numExperts * hidden, inter));    // [E·hidden, ff] = per-expert [hidden,ff]
                    for (int e = 0; e < numExperts; e++)
                    {
                        ((Dictionary<string, Tensor>)w)[$"blk.{i}.ffn_up.{e}.weight"] = SliceRows(up, e * inter, inter);
                        ((Dictionary<string, Tensor>)w)[$"blk.{i}.ffn_down.{e}.weight"] = SliceRows(dn, e * hidden, hidden);
                    }
                }
            }
            string[]? vocab = m.GetStringArray("tokenizer.ggml.tokens");
            bool hasRerankHead = w.ContainsKey("cls.weight") && w.ContainsKey("cls.output.weight");
            return new BertEmbeddingModel(handle, w, hidden, layers, heads, inter, maxPos, eps, pooling, vocab, useRope, ropeBase, useGeGlu, hasRerankHead, gateIsGelu, useAlibi, alibiSlopes, isNeoBert, moeEveryN, numExperts, expertsUsed);
        }
        catch { handle.Dispose(); throw; }
    }

    private Tensor W(string key) => _w[key];
    private Tensor? Wopt(string key) => _w.TryGetValue(key, out Tensor? t) ? t : null;

    /// <summary>Encodes WordPiece token ids (including <c>[CLS]</c>/<c>[SEP]</c>) into a pooled, L2-normalized sentence embedding of length <see cref="Hidden"/>.</summary>
    public float[] Encode(IBackend backend, IReadOnlyList<int> ids)
    {
        int hidden = Hidden;
        using Tensor h = RunEncoder(backend, ids);
        float* hp = (float*)h.DataPointer; backend.Sync();
        int seq = ids.Count;
        float[] outv = new float[hidden];
        if (PoolingType == 1)   // mean over tokens
        {
            for (int s = 0; s < seq; s++) for (int c = 0; c < hidden; c++) outv[c] += hp[(long)s * hidden + c];
            for (int c = 0; c < hidden; c++) outv[c] /= seq;
        }
        else                     // CLS (token 0); default
        {
            for (int c = 0; c < hidden; c++) outv[c] = hp[c];
        }
        double norm = 0; for (int c = 0; c < hidden; c++) norm += (double)outv[c] * outv[c];
        float inv = (float)(1.0 / Math.Sqrt(norm + 1e-12));
        for (int c = 0; c < hidden; c++) outv[c] *= inv;
        return outv;
    }

    /// <summary>Reranker (cross-encoder) relevance score for a (query, document) token sequence: runs the encoder, takes the CLS token (0), and applies the classification head (<c>cls</c> dense → tanh → <c>cls.output</c> → 1 logit); higher = more relevant. Requires <see cref="HasRerankHead"/>.</summary>
    public float Score(IBackend backend, IReadOnlyList<int> ids)
    {
        if (!HasRerankHead) throw new InvalidOperationException("This GGUF has no reranker head (cls.*).");
        int hidden = Hidden;
        using Tensor h = RunEncoder(backend, ids);
        // CLS token (0) → dense → tanh → out_proj → scalar logit.
        using Tensor cls = new(new TensorShape(1, 1, hidden), DType.F32);
        backend.Sync();
        Buffer.MemoryCopy((void*)h.DataPointer, (void*)cls.DataPointer, (long)hidden * 4, (long)hidden * 4);
        using Tensor dense = new(new TensorShape(1, 1, hidden), DType.F32);
        backend.Linear(dense, cls, W("cls.weight"), Wopt("cls.bias"));
        backend.Tanh(dense, dense);
        int outDim = (int)W("cls.output.weight").Shape[0];
        using Tensor logit = new(new TensorShape(1, 1, outDim), DType.F32);
        backend.Linear(logit, dense, W("cls.output.weight"), Wopt("cls.output.bias"));
        backend.Sync();
        return ((float*)logit.DataPointer)[0];
    }

    /// <summary>Runs the embeddings + all encoder blocks, returning the final hidden state <c>[1, seq, hidden]</c>.</summary>
    private Tensor RunEncoder(IBackend backend, IReadOnlyList<int> ids)
    {
        int seq = ids.Count, hidden = Hidden;
        TensorShape flat = new(1, seq, hidden);

        // 1. Embeddings: token + (absolute position, unless rotary/ALiBi) + token-type(segment 0). BERT-family then
        //    applies the embedding LayerNorm; neo-bert has neither token-type nor embedding norm (its first block's
        //    pre-norm handles it).
        Tensor emb = new(flat, DType.F32);
        float* e = (float*)emb.DataPointer;
        float* tok = (float*)W("token_embd.weight").DataPointer;       // [vocab, hidden]
        float* pos = (UseRope || UseAlibi) ? null : (float*)W("position_embd.weight").DataPointer;
        Tensor? typT = Wopt("token_types.weight");   // [2, hidden]; absent on neo-bert
        float* typ = typT is null ? null : (float*)typT.DataPointer;
        for (int s = 0; s < seq; s++)
        {
            float* dst = e + (long)s * hidden;
            float* tsrc = tok + (long)ids[s] * hidden;
            float* psrc = pos is null ? null : pos + (long)s * hidden;
            for (int c = 0; c < hidden; c++) dst[c] = tsrc[c] + (psrc is null ? 0f : psrc[c]) + (typ is null ? 0f : typ[c]);
        }
        Tensor normed;
        if (PreNormRms) { normed = emb; }   // neo-bert: no embedding norm; pre-norm blocks normalize internally
        else
        {
            normed = new(flat, DType.F32);
            backend.LayerNorm(normed, emb, W("token_embd_norm.weight"), W("token_embd_norm.bias"), Eps);
            emb.Dispose();
        }

        (float[]? cos, float[]? sin) = UseRope ? BuildRope(seq) : (null, null);

        Tensor h = normed;
        for (int i = 0; i < NumLayers; i++)
        {
            Tensor next = PreNormRms ? NeoBlock(backend, h, i, seq, cos, sin) : Block(backend, h, i, seq, cos, sin);
            h.Dispose(); h = next;
        }
        if (PreNormRms)   // neo-bert: a single final RMSNorm over the encoder output
        {
            Tensor fin = new(flat, DType.F32);
            backend.RmsNorm(fin, h, W("enc.output_norm.weight"), Eps);
            h.Dispose(); h = fin;
        }
        return h;
    }

    /// <summary>nomic-bert-moe routed FFN: a softmax router (<c>ffn_gate_inp</c>) selects the top-K of <see cref="NumExperts"/> non-gated GELU experts per token, blending their <c>down(gelu(up(x)))</c> outputs by the (unrenormalized) softmax weight.</summary>
    private Tensor MoeFfn(IBackend backend, Tensor normed1, int i, int seq)
    {
        int hidden = Hidden, inter = Intermediate, E = NumExperts, K = ExpertsUsed;
        string p = $"blk.{i}";
        TensorShape flat = new(1, seq, hidden);

        Tensor logits = new(new TensorShape(1, seq, E), DType.F32);
        backend.Linear(logits, normed1, W($"{p}.ffn_gate_inp.weight"), null);
        backend.Sync();
        float* lp = (float*)logits.DataPointer;
        // Per-token softmax over all experts → top-K → renormalize the kept weights to sum 1.
        float[] wts = new float[seq * E];
        for (int s = 0; s < seq; s++)
        {
            float mx = float.NegativeInfinity;
            for (int e = 0; e < E; e++) mx = MathF.Max(mx, lp[s * E + e]);
            float sum = 0; float[] sm = new float[E];
            for (int e = 0; e < E; e++) { sm[e] = MathF.Exp(lp[s * E + e] - mx); sum += sm[e]; }
            for (int e = 0; e < E; e++) sm[e] /= sum;
            // nomic-bert-moe uses the RAW softmax weights for the top-k (llama.cpp build_moe_ffn norm_w=false) —
            // the kept weights are NOT renormalized to sum 1.
            int[] top = Enumerable.Range(0, E).OrderByDescending(e => sm[e]).Take(K).ToArray();
            foreach (int e in top) wts[s * E + e] = sm[e];
        }
        logits.Dispose();

        float[] acc = new float[seq * hidden];
        for (int e = 0; e < E; e++)
        {
            bool any = false; for (int s = 0; s < seq; s++) if (wts[s * E + e] != 0f) { any = true; break; }
            if (!any) continue;
            Tensor up = new(new TensorShape(1, seq, inter), DType.F32);
            backend.Linear(up, normed1, W($"{p}.ffn_up.{e}.weight"), null);
            backend.Gelu(up, up);
            Tensor de = new(flat, DType.F32);
            backend.Linear(de, up, W($"{p}.ffn_down.{e}.weight"), null);
            backend.Sync();
            float* dp = (float*)de.DataPointer;
            for (int s = 0; s < seq; s++)
            {
                float wv = wts[s * E + e]; if (wv == 0f) continue;
                for (int c = 0; c < hidden; c++) acc[s * hidden + c] += wv * dp[s * hidden + c];
            }
            up.Dispose(); de.Dispose();
        }
        Tensor outp = new(flat, DType.F32);
        float* op = (float*)outp.DataPointer;
        for (int x = 0; x < seq * hidden; x++) op[x] = acc[x];
        return outp;
    }

    /// <summary>One neo-bert pre-norm block: RMSNorm → RoPE self-attn (bidirectional) → +res → RMSNorm → SwiGLU FFN → +res; no biases, fused QKV and SwiGLU gate/up split at load.</summary>
    private Tensor NeoBlock(IBackend backend, Tensor x, int i, int seq, float[]? cos, float[]? sin)
    {
        int hidden = Hidden, heads = NumHeads, hd = HeadDim;
        string p = $"blk.{i}";
        TensorShape flat = new(1, seq, hidden);

        Tensor n1 = new(flat, DType.F32);
        backend.RmsNorm(n1, x, W($"{p}.attn_norm.weight"), Eps);

        Tensor q = new(new TensorShape(1, seq, heads, hd), DType.F32);
        Tensor k = new(new TensorShape(1, seq, heads, hd), DType.F32);
        Tensor v = new(new TensorShape(1, seq, heads, hd), DType.F32);
        backend.Linear(q, n1, W($"{p}.attn_q.weight"), null);
        backend.Linear(k, n1, W($"{p}.attn_k.weight"), null);
        backend.Linear(v, n1, W($"{p}.attn_v.weight"), null);
        n1.Dispose();
        if (UseRope)
        {
            // neo-bert uses the interleaved (complex view_as_complex) RoPE — pairs (2i, 2i+1) — not NeoX rotate-half.
            backend.Sync();
            ApplyRopeInterleaved(q, RopeBase, seq, heads, hd);
            ApplyRopeInterleaved(k, RopeBase, seq, heads, hd);
        }

        Tensor qM = new(new TensorShape(1, heads, seq, hd), DType.F32);
        Tensor kM = new(new TensorShape(1, heads, seq, hd), DType.F32);
        Tensor vM = new(new TensorShape(1, heads, seq, hd), DType.F32);
        backend.Permute0213(qM, q, seq, heads, hd);
        backend.Permute0213(kM, k, seq, heads, hd);
        backend.Permute0213(vM, v, seq, heads, hd);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor attn = new(new TensorShape(1, heads, seq, hd), DType.F32);
        backend.FlashAttention(attn, qM, kM, vM, seq, 1, causal: false, qOffset: 0, 1f / MathF.Sqrt(hd));
        qM.Dispose(); kM.Dispose(); vM.Dispose();
        Tensor attnFlat = new(flat, DType.F32);
        backend.Permute0213(attnFlat, attn, heads, seq, hd);
        attn.Dispose();
        Tensor attnOut = new(flat, DType.F32);
        backend.Linear(attnOut, attnFlat, W($"{p}.attn_output.weight"), null);
        attnFlat.Dispose();

        Tensor afterAttn = new(flat, DType.F32);
        backend.Add(afterAttn, x, attnOut);
        attnOut.Dispose();

        Tensor n2 = new(flat, DType.F32);
        backend.RmsNorm(n2, afterAttn, W($"{p}.ffn_norm.weight"), Eps);
        Tensor act = new(new TensorShape(1, seq, Intermediate), DType.F32);
        Tensor up = new(new TensorShape(1, seq, Intermediate), DType.F32);
        Tensor gate = new(new TensorShape(1, seq, Intermediate), DType.F32);
        backend.Linear(up, n2, W($"{p}.ffn_up.weight"), null);
        backend.Linear(gate, n2, W($"{p}.ffn_gate.weight"), null);
        n2.Dispose();
        backend.Silu(gate, gate);
        backend.Mul(act, gate, up);
        gate.Dispose(); up.Dispose();
        Tensor down = new(flat, DType.F32);
        backend.Linear(down, act, W($"{p}.ffn_down.weight"), null);
        act.Dispose();
        Tensor result = new(flat, DType.F32);
        backend.Add(result, afterAttn, down);
        afterAttn.Dispose(); down.Dispose();
        return result;
    }

    /// <summary>One post-norm block: (optionally rotary) bidirectional self-attention → +res → LayerNorm → FFN (GELU or GeGLU) → +res → LayerNorm; biases are optional (nomic-bert has none on the linears).</summary>
    private Tensor Block(IBackend backend, Tensor x, int i, int seq, float[]? cos, float[]? sin)
    {
        int hidden = Hidden, heads = NumHeads, hd = HeadDim;
        string p = $"blk.{i}";
        TensorShape flat = new(1, seq, hidden);

        Tensor q = new(new TensorShape(1, seq, heads, hd), DType.F32);
        Tensor k = new(new TensorShape(1, seq, heads, hd), DType.F32);
        Tensor v = new(new TensorShape(1, seq, heads, hd), DType.F32);
        backend.Linear(q, x, W($"{p}.attn_q.weight"), Wopt($"{p}.attn_q.bias"));
        backend.Linear(k, x, W($"{p}.attn_k.weight"), Wopt($"{p}.attn_k.bias"));
        backend.Linear(v, x, W($"{p}.attn_v.weight"), Wopt($"{p}.attn_v.bias"));
        if (UseRope)
        {
            backend.Sync();
            ApplyRope(q, cos!, sin!, seq, heads, hd);
            ApplyRope(k, cos!, sin!, seq, heads, hd);
        }

        Tensor qM = new(new TensorShape(1, heads, seq, hd), DType.F32);
        Tensor kM = new(new TensorShape(1, heads, seq, hd), DType.F32);
        Tensor vM = new(new TensorShape(1, heads, seq, hd), DType.F32);
        backend.Permute0213(qM, q, seq, heads, hd);
        backend.Permute0213(kM, k, seq, heads, hd);
        backend.Permute0213(vM, v, seq, heads, hd);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor attn = new(new TensorShape(1, heads, seq, hd), DType.F32);
        backend.FlashAttention(attn, qM, kM, vM, seq, 1, causal: false, qOffset: 0, 1f / MathF.Sqrt(hd),
            softcap: 0f, sink: null, slidingWindow: 0, alibiSlopes: _alibiSlopes);
        qM.Dispose(); kM.Dispose(); vM.Dispose();
        Tensor attnFlat = new(flat, DType.F32);
        backend.Permute0213(attnFlat, attn, heads, seq, hd);
        attn.Dispose();
        Tensor attnOut = new(flat, DType.F32);
        backend.Linear(attnOut, attnFlat, W($"{p}.attn_output.weight"), Wopt($"{p}.attn_output.bias"));
        attnFlat.Dispose();

        Tensor afterAttn = new(flat, DType.F32);
        backend.Add(afterAttn, x, attnOut);
        attnOut.Dispose();
        Tensor normed1 = new(flat, DType.F32);
        backend.LayerNorm(normed1, afterAttn, W($"{p}.attn_output_norm.weight"), W($"{p}.attn_output_norm.bias"), Eps);
        afterAttn.Dispose();

        // FFN: routed MoE on MoE layers (nomic-bert-moe), else dense — GeGLU (down(act(gate)·up)) or plain GELU.
        Tensor down;
        if (IsMoeLayer(i)) down = MoeFfn(backend, normed1, i, seq);
        else
        {
            Tensor act = new(new TensorShape(1, seq, Intermediate), DType.F32);
            Tensor up = new(new TensorShape(1, seq, Intermediate), DType.F32);
            backend.Linear(up, normed1, W($"{p}.ffn_up.weight"), Wopt($"{p}.ffn_up.bias"));
            if (UseGeGlu)
            {
                // Gated MLP: nomic-bert = SwiGLU (SiLU gate); jina-bert-v2 = GEGLU (GELU gate). down(act(gate)·up).
                Tensor gate = new(new TensorShape(1, seq, Intermediate), DType.F32);
                backend.Linear(gate, normed1, W($"{p}.ffn_gate.weight"), Wopt($"{p}.ffn_gate.bias"));
                if (GateIsGelu) backend.Gelu(gate, gate); else backend.Silu(gate, gate);
                backend.Mul(act, gate, up);
                gate.Dispose();
            }
            else backend.Gelu(act, up);
            up.Dispose();
            down = new(flat, DType.F32);
            backend.Linear(down, act, W($"{p}.ffn_down.weight"), Wopt($"{p}.ffn_down.bias"));
            act.Dispose();
        }
        Tensor afterFfn = new(flat, DType.F32);
        backend.Add(afterFfn, normed1, down);
        normed1.Dispose(); down.Dispose();
        Tensor result = new(flat, DType.F32);
        backend.LayerNorm(result, afterFfn, W($"{p}.layer_output_norm.weight"), W($"{p}.layer_output_norm.bias"), Eps);
        afterFfn.Dispose();
        return result;
    }

    /// <summary>Standard rotary cos/sin tables [seq, headDim] (NeoX rotate-half, base <see cref="RopeBase"/>).</summary>
    private (float[] cos, float[] sin) BuildRope(int seq)
    {
        int hd = HeadDim, half = hd / 2;
        float[] cos = new float[seq * hd], sin = new float[seq * hd];
        for (int p = 0; p < seq; p++)
            for (int i = 0; i < half; i++)
            {
                float ang = p / MathF.Pow(RopeBase, (2f * i) / hd);
                float c = MathF.Cos(ang), s = MathF.Sin(ang);
                cos[p * hd + i] = c; cos[p * hd + i + half] = c;
                sin[p * hd + i] = s; sin[p * hd + i + half] = s;
            }
        return (cos, sin);
    }

    /// <summary>Interleaved (GPT-J / complex) RoPE: rotates each adjacent pair (2i, 2i+1) by pos·θ_i, θ_i = base^(−2i/hd); matches NeoBERT's <c>view_as_complex(x.reshape(...,-1,2))</c> rotary, distinct from the NeoX rotate-half in <see cref="ApplyRope"/>.</summary>
    private static void ApplyRopeInterleaved(Tensor t, float ropeBase, int seq, int heads, int hd)
    {
        float* pp = (float*)t.DataPointer;   // [1, seq, heads, hd]
        int half = hd / 2;
        for (int s = 0; s < seq; s++)
            for (int h = 0; h < heads; h++)
            {
                float* row = pp + ((long)s * heads + h) * hd;
                for (int i = 0; i < half; i++)
                {
                    float ang = s / MathF.Pow(ropeBase, (2f * i) / hd);
                    float c = MathF.Cos(ang), sn = MathF.Sin(ang);
                    float x0 = row[2 * i], x1 = row[2 * i + 1];
                    row[2 * i] = x0 * c - x1 * sn;
                    row[2 * i + 1] = x0 * sn + x1 * c;
                }
            }
    }

    private static void ApplyRope(Tensor t, float[] cos, float[] sin, int seq, int heads, int hd)
    {
        float* pp = (float*)t.DataPointer;   // [1, seq, heads, hd]
        int half = hd / 2;
        float[] tmp = new float[hd];
        for (int s = 0; s < seq; s++)
            for (int h = 0; h < heads; h++)
            {
                float* row = pp + ((long)s * heads + h) * hd;
                for (int e = 0; e < hd; e++) tmp[e] = row[e];
                for (int e = 0; e < hd; e++)
                {
                    float rot = e < half ? -tmp[e + half] : tmp[e - half];
                    row[e] = tmp[e] * cos[s * hd + e] + rot * sin[s * hd + e];
                }
            }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _handle.Dispose();
    }
}
