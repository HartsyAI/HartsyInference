using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.Gguf;

namespace HartsyInference.LLM.Embeddings;

/// <summary>A BERT-family text-embedding model loaded from a llama.cpp <c>bert</c> GGUF (bge, all-MiniLM, nomic,
/// e5, …). Runs the bidirectional post-norm encoder, pools the token states (CLS / mean per the GGUF
/// <c>bert.pooling_type</c>), and L2-normalizes — producing a sentence embedding. Reuses the engine's
/// Linear / LayerNorm / FlashAttention (non-causal) primitives.</summary>
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
    /// <summary>The model's WordPiece vocabulary (from <c>tokenizer.ggml.tokens</c>), or null if absent.</summary>
    public string[]? Vocab { get; }

    private BertEmbeddingModel(GgufModelLoader.LoadedGgufModel handle, IReadOnlyDictionary<string, Tensor> w,
        int hidden, int layers, int heads, int inter, int maxPos, float eps, int pooling, string[]? vocab)
    {
        _handle = handle; _w = w;
        Hidden = hidden; NumLayers = layers; NumHeads = heads; HeadDim = hidden / heads; Intermediate = inter;
        MaxPos = maxPos; Eps = eps <= 0f ? 1e-12f : eps; PoolingType = pooling; Vocab = vocab;
    }

    private static Tensor Relabel(Tensor t)
    {
        Tensor outp = new(new TensorShape((int)t.Shape[1], (int)t.Shape[0]), DType.F32);
        Buffer.MemoryCopy((void*)t.DataPointer, (void*)outp.DataPointer, outp.ElementCount * 4, t.ElementCount * 4);
        return outp;
    }

    public static BertEmbeddingModel Load(string ggufPath)
    {
        (Dictionary<string, Tensor> w, GgufModelLoader.LoadedGgufModel handle) = GgufModelLoader.LoadDequantized(ggufPath, DType.F32);
        try
        {
            // nn.Linear weights are ggml ne=[in,out] with raw bytes already row-major [out,in] → relabel for Linear.
            foreach (string key in w.Keys.ToList())
            {
                if (!key.EndsWith(".weight", StringComparison.Ordinal) || w[key].Shape.Rank != 2) continue;
                if (key.Contains("attn_q") || key.Contains("attn_k") || key.Contains("attn_v") || key.Contains("attn_output")
                    || key.Contains("ffn_up") || key.Contains("ffn_down"))
                    w[key] = Relabel(w[key]);
            }
            if (Environment.GetEnvironmentVariable("HARTSY_BERT_DEBUG") == "1")
                Console.Error.WriteLine("[bert keys] " + string.Join(" ", w.Keys.Where(k => !k.StartsWith("blk.")).OrderBy(k => k)));
            GgufMetadata m = handle.Metadata;
            int hidden = (int)m.GetUInt32("bert.embedding_length");
            int layers = (int)m.GetUInt32("bert.block_count");
            int heads = (int)m.GetUInt32("bert.attention.head_count");
            int inter = (int)m.GetUInt32("bert.feed_forward_length");
            int maxPos = (int)m.GetUInt32("bert.context_length", 512u);
            float eps = m.GetFloat32("bert.attention.layer_norm_epsilon", 1e-12f);
            int pooling = (int)m.GetUInt32("bert.pooling_type", 2u);
            string[]? vocab = m.GetStringArray("tokenizer.ggml.tokens");
            return new BertEmbeddingModel(handle, w, hidden, layers, heads, inter, maxPos, eps, pooling, vocab);
        }
        catch { handle.Dispose(); throw; }
    }

    private Tensor W(string key) => _w[key];

    /// <summary>Encodes WordPiece token ids (including <c>[CLS]</c>/<c>[SEP]</c>) into a pooled, L2-normalized
    /// sentence embedding of length <see cref="Hidden"/>.</summary>
    public float[] Encode(IBackend backend, IReadOnlyList<int> ids)
    {
        int seq = ids.Count, hidden = Hidden, heads = NumHeads, hd = HeadDim;
        TensorShape flat = new(1, seq, hidden);

        // 1. Embeddings: token + absolute position + token-type(segment 0), then LayerNorm.
        Tensor emb = new(flat, DType.F32);
        float* e = (float*)emb.DataPointer;
        float* tok = (float*)W("token_embd.weight").DataPointer;       // [vocab, hidden]
        float* pos = (float*)W("position_embd.weight").DataPointer;    // [maxPos, hidden]
        float* typ = (float*)W("token_types.weight").DataPointer;      // [2, hidden]
        for (int s = 0; s < seq; s++)
        {
            float* dst = e + (long)s * hidden;
            float* tsrc = tok + (long)ids[s] * hidden;
            float* psrc = pos + (long)s * hidden;
            for (int c = 0; c < hidden; c++) dst[c] = tsrc[c] + psrc[c] + typ[c];   // type[0]
        }
        Tensor normed = new(flat, DType.F32);
        backend.LayerNorm(normed, emb, W("token_embd_norm.weight"), W("token_embd_norm.bias"), Eps);
        emb.Dispose();

        Tensor h = normed;
        for (int i = 0; i < NumLayers; i++)
        {
            Tensor next = Block(backend, h, i, seq);
            h.Dispose(); h = next;
        }

        // Pool + L2 normalize.
        float* hp = (float*)h.DataPointer;   // [seq, hidden] (D2H sync)
        backend.Sync();
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
        h.Dispose();
        double norm = 0; for (int c = 0; c < hidden; c++) norm += (double)outv[c] * outv[c];
        float inv = (float)(1.0 / Math.Sqrt(norm + 1e-12));
        for (int c = 0; c < hidden; c++) outv[c] *= inv;
        return outv;
    }

    /// <summary>One post-norm BERT block: bidirectional self-attention → +res → LayerNorm → FFN(GELU) → +res → LayerNorm.</summary>
    private Tensor Block(IBackend backend, Tensor x, int i, int seq)
    {
        int hidden = Hidden, heads = NumHeads, hd = HeadDim;
        string p = $"blk.{i}";
        TensorShape flat = new(1, seq, hidden);

        Tensor q = new(new TensorShape(1, seq, heads, hd), DType.F32);
        Tensor k = new(new TensorShape(1, seq, heads, hd), DType.F32);
        Tensor v = new(new TensorShape(1, seq, heads, hd), DType.F32);
        backend.Linear(q, x, W($"{p}.attn_q.weight"), W($"{p}.attn_q.bias"));
        backend.Linear(k, x, W($"{p}.attn_k.weight"), W($"{p}.attn_k.bias"));
        backend.Linear(v, x, W($"{p}.attn_v.weight"), W($"{p}.attn_v.bias"));

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
        backend.Linear(attnOut, attnFlat, W($"{p}.attn_output.weight"), W($"{p}.attn_output.bias"));
        attnFlat.Dispose();

        Tensor afterAttn = new(flat, DType.F32);
        backend.Add(afterAttn, x, attnOut);
        attnOut.Dispose();
        Tensor normed1 = new(flat, DType.F32);
        backend.LayerNorm(normed1, afterAttn, W($"{p}.attn_output_norm.weight"), W($"{p}.attn_output_norm.bias"), Eps);
        afterAttn.Dispose();

        Tensor up = new(new TensorShape(1, seq, Intermediate), DType.F32);
        backend.Linear(up, normed1, W($"{p}.ffn_up.weight"), W($"{p}.ffn_up.bias"));
        Tensor act = new(new TensorShape(1, seq, Intermediate), DType.F32);
        backend.Gelu(act, up);
        up.Dispose();
        Tensor down = new(flat, DType.F32);
        backend.Linear(down, act, W($"{p}.ffn_down.weight"), W($"{p}.ffn_down.bias"));
        act.Dispose();
        Tensor afterFfn = new(flat, DType.F32);
        backend.Add(afterFfn, normed1, down);
        normed1.Dispose(); down.Dispose();
        Tensor result = new(flat, DType.F32);
        backend.LayerNorm(result, afterFfn, W($"{p}.layer_output_norm.weight"), W($"{p}.layer_output_norm.bias"), Eps);
        afterFfn.Dispose();
        return result;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _handle.Dispose();
    }
}
