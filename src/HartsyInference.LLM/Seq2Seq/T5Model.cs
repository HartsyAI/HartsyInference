using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.LLM.Seq2Seq;

/// <summary>T5 / FLAN-T5 encoder-decoder (seq2seq) loaded from a llama.cpp <c>t5</c> GGUF. The encoder runs once
/// over the input; the decoder autoregressively cross-attends to the encoder output. T5 quirks handled:
/// inner attention dim = n_heads·key_length (≠ d_model); <b>no 1/√d attention scaling</b>; learned <b>relative
/// position bias</b> (bucketed, shared across layers — weights on block 0; bidirectional for the encoder,
/// causal/unidirectional for the decoder self-attn; none for cross-attn); T5LayerNorm = RMSNorm (no bias);
/// GeGLU FFN (FLAN). Attention reuses <see cref="IBackend.ScaledDotProductAttention"/> with <c>scale = 1</c> and an
/// additive per-head mask carrying the relative-position bias (+ causal −∞ for the decoder self-attn).</summary>
public sealed unsafe class T5Model : IDisposable
{
    private readonly GgufModelLoader.LoadedGgufModel _handle;
    private readonly IReadOnlyDictionary<string, Tensor> _w;
    private int _disposed;

    public int DModel { get; }
    public int NumLayers { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }      // key_length
    public int Inner { get; }        // n_heads * key_length
    public int Intermediate { get; }
    public int VocabSize { get; }
    public int NumBuckets { get; }
    public int DecoderStartToken { get; }
    public float Eps { get; }
    private const int MaxDistance = 128;

    private T5Model(GgufModelLoader.LoadedGgufModel handle, IReadOnlyDictionary<string, Tensor> w,
        int dModel, int layers, int heads, int headDim, int inter, int vocab, int buckets, int decStart, float eps)
    {
        _handle = handle; _w = w;
        DModel = dModel; NumLayers = layers; NumHeads = heads; HeadDim = headDim; Inner = heads * headDim;
        Intermediate = inter; VocabSize = vocab; NumBuckets = buckets; DecoderStartToken = decStart; Eps = eps <= 0f ? 1e-6f : eps;
    }

    private static Tensor Relabel(Tensor t)
    {
        Tensor outp = new(new TensorShape((int)t.Shape[1], (int)t.Shape[0]), DType.F32);
        Buffer.MemoryCopy((void*)t.DataPointer, (void*)outp.DataPointer, outp.ElementCount * 4, t.ElementCount * 4);
        return outp;
    }

    public static T5Model Load(string ggufPath)
    {
        (Dictionary<string, Tensor> w, GgufModelLoader.LoadedGgufModel handle) = GgufModelLoader.LoadDequantized(ggufPath, DType.F32);
        try
        {
            foreach (string key in w.Keys.ToList())
            {
                if (!key.EndsWith(".weight", StringComparison.Ordinal) || w[key].Shape.Rank != 2) continue;
                if (key.Contains("attn_q") || key.Contains("attn_k") || key.Contains("attn_v") || key.Contains("attn_o")
                    || key.Contains("cross_attn_q") || key.Contains("cross_attn_k") || key.Contains("cross_attn_v") || key.Contains("cross_attn_o")
                    || key.Contains("ffn_up") || key.Contains("ffn_down") || key.Contains("ffn_gate")
                    || key == "output.weight" || key == "token_embd.weight")
                    w[key] = Relabel(w[key]);
            }
            GgufMetadata m = handle.Metadata;
            int dModel = (int)m.GetUInt32("t5.embedding_length");
            int layers = (int)m.GetUInt32("t5.block_count");
            int heads = (int)m.GetUInt32("t5.attention.head_count");
            int headDim = (int)m.GetUInt32("t5.attention.key_length", 64u);
            int inter = (int)m.GetUInt32("t5.feed_forward_length");
            int buckets = (int)m.GetUInt32("t5.attention.relative_buckets_count", 32u);
            int decStart = (int)m.GetUInt32("t5.decoder_start_token_id", 0u);
            float eps = m.GetFloat32("t5.attention.layer_norm_rms_epsilon", 1e-6f);
            int vocab = (int)w["token_embd.weight"].Shape[0];
            return new T5Model(handle, w, dModel, layers, heads, headDim, inter, vocab, buckets, decStart, eps);
        }
        catch { handle.Dispose(); throw; }
    }

    private Tensor W(string key) => _w[key];

    // ── T5 relative-position bucketing (matches HF _relative_position_bucket) ──
    private int Bucket(int relPos, bool bidirectional)
    {
        int nb = NumBuckets, ret = 0;
        if (bidirectional)
        {
            nb /= 2;
            if (relPos > 0) ret += nb;
            relPos = Math.Abs(relPos);
        }
        else relPos = -Math.Min(relPos, 0);
        int maxExact = nb / 2;
        if (relPos < maxExact) return ret + relPos;
        int large = maxExact + (int)(MathF.Log(relPos / (float)maxExact) / MathF.Log(MaxDistance / (float)maxExact) * (nb - maxExact));
        return ret + Math.Min(large, nb - 1);
    }

    /// <summary>Builds the additive per-head attention mask [1, heads, qLen, kLen] = relative-position bias
    /// (+ causal −∞ when <paramref name="causal"/>). <paramref name="relB"/> is the [buckets, heads] bias table.</summary>
    private Tensor RelBiasMask(int qLen, int kLen, bool bidirectional, bool causal, Tensor relB)
    {
        Tensor mask = new(new TensorShape(1, NumHeads, qLen, kLen), DType.F32);
        float* mp = (float*)mask.DataPointer;
        float* rb = (float*)relB.DataPointer;   // [buckets, heads]
        for (int h = 0; h < NumHeads; h++)
            for (int i = 0; i < qLen; i++)
                for (int j = 0; j < kLen; j++)
                {
                    long idx = ((long)h * qLen + i) * kLen + j;
                    if (causal && j > i) { mp[idx] = -1e30f; continue; }
                    int b = Bucket(j - i, bidirectional);
                    mp[idx] = rb[(long)b * NumHeads + h];
                }
        return mask;
    }

    /// <summary>Builds a zero additive mask [1, heads, qLen, kLen] (cross-attention: no bias, no causal).</summary>
    private Tensor ZeroMask(int qLen, int kLen)
    {
        Tensor mask = new(new TensorShape(1, NumHeads, qLen, kLen), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (long n = 0; n < (long)NumHeads * qLen * kLen; n++) mp[n] = 0f;
        return mask;
    }

    /// <summary>Multi-head attention with T5 conventions: no 1/√d scaling, additive per-head mask carrying the
    /// relative-position bias. <paramref name="kvSrc"/> = the source of K/V (self = x; cross = encoder output).</summary>
    private Tensor Attention(IBackend backend, Tensor x, Tensor kvSrc, int qLen, int kLen, string prefix, Tensor mask)
    {
        int heads = NumHeads, hd = HeadDim, inner = Inner, d = DModel;
        Tensor q = new(new TensorShape(1, qLen, heads, hd), DType.F32);
        Tensor k = new(new TensorShape(1, kLen, heads, hd), DType.F32);
        Tensor v = new(new TensorShape(1, kLen, heads, hd), DType.F32);
        backend.Linear(q, x, W($"{prefix}_q.weight"), null);
        backend.Linear(k, kvSrc, W($"{prefix}_k.weight"), null);
        backend.Linear(v, kvSrc, W($"{prefix}_v.weight"), null);
        Tensor qM = new(new TensorShape(1, heads, qLen, hd), DType.F32);
        Tensor kM = new(new TensorShape(1, heads, kLen, hd), DType.F32);
        Tensor vM = new(new TensorShape(1, heads, kLen, hd), DType.F32);
        backend.Permute0213(qM, q, qLen, heads, hd);
        backend.Permute0213(kM, k, kLen, heads, hd);
        backend.Permute0213(vM, v, kLen, heads, hd);
        q.Dispose(); k.Dispose(); v.Dispose();
        Tensor attn = new(new TensorShape(1, heads, qLen, hd), DType.F32);
        backend.ScaledDotProductAttention(attn, qM, kM, vM, mask, 1f);   // T5: scale = 1 (no 1/√d)
        qM.Dispose(); kM.Dispose(); vM.Dispose();
        Tensor flat = new(new TensorShape(1, qLen, inner), DType.F32);
        backend.Permute0213(flat, attn, heads, qLen, hd);
        attn.Dispose();
        Tensor o = new(new TensorShape(1, qLen, d), DType.F32);
        backend.Linear(o, flat, W($"{prefix}_o.weight"), null);
        flat.Dispose();
        return o;
    }

    private Tensor Ffn(IBackend backend, Tensor x, int len, string p)
    {
        Tensor gate = new(new TensorShape(1, len, Intermediate), DType.F32);
        backend.Linear(gate, x, W($"{p}.ffn_gate.weight"), null);
        backend.Gelu(gate, gate);
        Tensor up = new(new TensorShape(1, len, Intermediate), DType.F32);
        backend.Linear(up, x, W($"{p}.ffn_up.weight"), null);
        Tensor gu = new(new TensorShape(1, len, Intermediate), DType.F32);
        backend.Mul(gu, gate, up); gate.Dispose(); up.Dispose();
        Tensor down = new(new TensorShape(1, len, DModel), DType.F32);
        backend.Linear(down, gu, W($"{p}.ffn_down.weight"), null); gu.Dispose();
        return down;
    }

    private Tensor Embed(IReadOnlyList<int> ids)
    {
        int n = ids.Count, d = DModel;
        Tensor h = new(new TensorShape(1, n, d), DType.F32);
        float* hp = (float*)h.DataPointer;
        float* emb = (float*)W("token_embd.weight").DataPointer;
        for (int s = 0; s < n; s++) Buffer.MemoryCopy(emb + (long)ids[s] * d, hp + (long)s * d, (long)d * 4, (long)d * 4);
        return h;
    }

    /// <summary>Encodes the input token ids → encoder hidden state <c>[1, seqIn, dModel]</c>.</summary>
    public Tensor Encode(IBackend backend, IReadOnlyList<int> ids)
    {
        int seq = ids.Count, d = DModel;
        Tensor h = Embed(ids);
        using Tensor mask = RelBiasMask(seq, seq, bidirectional: true, causal: false, W("enc.blk.0.attn_rel_b.weight"));
        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"enc.blk.{i}";
            Tensor xn = new(new TensorShape(1, seq, d), DType.F32);
            backend.RmsNorm(xn, h, W($"{p}.attn_norm.weight"), Eps);
            Tensor o = Attention(backend, xn, xn, seq, seq, $"{p}.attn", mask);
            xn.Dispose();
            Tensor h2 = new(new TensorShape(1, seq, d), DType.F32);
            backend.Add(h2, h, o); o.Dispose(); h.Dispose(); h = h2;
            Tensor fn = new(new TensorShape(1, seq, d), DType.F32);
            backend.RmsNorm(fn, h, W($"{p}.ffn_norm.weight"), Eps);
            Tensor ff = Ffn(backend, fn, seq, p); fn.Dispose();
            Tensor h3 = new(new TensorShape(1, seq, d), DType.F32);
            backend.Add(h3, h, ff); ff.Dispose(); h.Dispose(); h = h3;
        }
        Tensor enc = new(new TensorShape(1, seq, d), DType.F32);
        backend.RmsNorm(enc, h, W("enc.output_norm.weight"), Eps);
        h.Dispose();
        return enc;
    }

    /// <summary>Runs the decoder over <paramref name="decIds"/> (teacher-forced) against the encoder output and
    /// returns the last-position logits.</summary>
    public float[] DecodeLogits(IBackend backend, Tensor enc, int encLen, IReadOnlyList<int> decIds)
    {
        int seq = decIds.Count, d = DModel;
        Tensor h = Embed(decIds);
        using Tensor selfMask = RelBiasMask(seq, seq, bidirectional: false, causal: true, W("dec.blk.0.attn_rel_b.weight"));
        using Tensor crossMask = ZeroMask(seq, encLen);
        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"dec.blk.{i}";
            Tensor xn = new(new TensorShape(1, seq, d), DType.F32);
            backend.RmsNorm(xn, h, W($"{p}.attn_norm.weight"), Eps);
            Tensor o = Attention(backend, xn, xn, seq, seq, $"{p}.attn", selfMask);
            xn.Dispose();
            Tensor h2 = new(new TensorShape(1, seq, d), DType.F32);
            backend.Add(h2, h, o); o.Dispose(); h.Dispose(); h = h2;
            // cross-attention to encoder output
            Tensor cn = new(new TensorShape(1, seq, d), DType.F32);
            backend.RmsNorm(cn, h, W($"{p}.cross_attn_norm.weight"), Eps);
            Tensor co = Attention(backend, cn, enc, seq, encLen, $"{p}.cross_attn", crossMask);
            cn.Dispose();
            Tensor h3 = new(new TensorShape(1, seq, d), DType.F32);
            backend.Add(h3, h, co); co.Dispose(); h.Dispose(); h = h3;
            Tensor fn = new(new TensorShape(1, seq, d), DType.F32);
            backend.RmsNorm(fn, h, W($"{p}.ffn_norm.weight"), Eps);
            Tensor ff = Ffn(backend, fn, seq, p); fn.Dispose();
            Tensor h4 = new(new TensorShape(1, seq, d), DType.F32);
            backend.Add(h4, h, ff); ff.Dispose(); h.Dispose(); h = h4;
        }
        Tensor dn = new(new TensorShape(1, seq, d), DType.F32);
        backend.RmsNorm(dn, h, W("dec.output_norm.weight"), Eps);
        h.Dispose();
        backend.Sync();
        using Tensor last = new(new TensorShape(1, 1, d), DType.F32);
        Buffer.MemoryCopy((byte*)dn.DataPointer + (long)(seq - 1) * d * 4, (void*)last.DataPointer, (long)d * 4, (long)d * 4);
        dn.Dispose();
        using Tensor logits = new(new TensorShape(1, 1, VocabSize), DType.F32);
        backend.Linear(logits, last, W("output.weight"), null);
        backend.Sync();
        float[] outv = new float[VocabSize];
        fixed (float* o = outv) Buffer.MemoryCopy((void*)logits.DataPointer, o, (long)VocabSize * 4, (long)VocabSize * 4);
        return outv;
    }

    /// <summary>Greedy seq2seq generation: encode the input, then decode from the start token until EOS (id 1).</summary>
    public List<int> Generate(IBackend backend, IReadOnlyList<int> inputIds, int maxNew = 64)
    {
        using Tensor enc = Encode(backend, inputIds);
        int encLen = inputIds.Count;
        List<int> dec = new() { DecoderStartToken };
        List<int> outTokens = new();
        for (int step = 0; step < maxNew; step++)
        {
            float[] logits = DecodeLogits(backend, enc, encLen, dec);
            int next = 0; for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[next]) next = i;
            if (next == 1) break;   // </s>
            outTokens.Add(next); dec.Add(next);
        }
        return outTokens;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _handle.Dispose();
    }
}
