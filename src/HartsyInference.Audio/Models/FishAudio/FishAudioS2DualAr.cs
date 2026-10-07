using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>Fish Audio S2 Dual-AR model, ported from fish-speech's <c>DualARTransformer</c>. A 36-layer slow
/// Qwen3-style transformer (per-head Q/K norm, interleaved RoPE, tied text embedding) reads the summed text +
/// codebook embedding of each frame and predicts the next semantic token; its POST-norm hidden state then drives a
/// 4-layer fast transformer that predicts the remaining nine codebooks. Both stacks run on the shared
/// <see cref="GenericTransformer"/>, so BF16 weights stay in their stored dtype.</summary>
public sealed unsafe class FishAudioS2DualAr : IDisposable
{
    private readonly FishAudioS2Config _cfg;
    private readonly GenericTransformer _slow;
    private readonly GenericTransformer _fast;
    private Tensor? _codebookEmb, _fastEmb, _fastOut;
    private int[]? _allowed;
    private int _disposed;

    public FishAudioS2DualAr(FishAudioS2Config cfg)
    {
        _cfg = cfg;
        _slow = new GenericTransformer(ToTransformerConfig(cfg.Slow, tieEmbeddings: true));
        _fast = new GenericTransformer(ToTransformerConfig(cfg.Fast, tieEmbeddings: false));
    }

    public int HiddenSize => _cfg.Slow.HiddenSize;

    private static TransformerConfig ToTransformerConfig(FishAudioTransformerConfig c, bool tieEmbeddings) => new()
    {
        HiddenSize = c.HiddenSize,
        NumLayers = c.NumHiddenLayers,
        NumHeads = c.NumAttentionHeads,
        NumKvHeads = c.NumKeyValueHeads,
        HeadDim = c.HeadDim > 0 ? c.HeadDim : c.HiddenSize / c.NumAttentionHeads,
        IntermediateSize = c.IntermediateSize,
        VocabSize = c.VocabSize,
        MaxPositionEmbeddings = c.MaxPositionEmbeddings,
        RopeTheta = c.RopeTheta,
        RmsNormEps = c.RmsNormEps,
        AttentionBias = false,
        QkNorm = c.QkNorm,
        TieWordEmbeddings = tieEmbeddings,
        Rope = RopeStyle.Interleaved,   // fish-speech's apply_rotary_emb pairs adjacent dims
    };

    /// <summary>Loads the checkpoint (<c>text_model.model.*</c>, <c>audio_decoder.*</c> keys, split from the fused
    /// <c>wqkv</c>) without converting weight dtype.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        Dictionary<string, Tensor> slow = RemapLayers(w, "text_model.model", "model", _cfg.Slow);
        slow["model.embed_tokens.weight"] = w["text_model.model.embeddings.weight"];
        slow["model.norm.weight"] = w["text_model.model.norm.weight"];
        _slow.LoadWeights(slow, "model");

        Dictionary<string, Tensor> fast = RemapLayers(w, "audio_decoder", "fast_model", _cfg.Fast);
        fast["fast_model.norm.weight"] = w["audio_decoder.norm.weight"];
        _fast.LoadWeightsHeadless(fast, "fast_model");

        _codebookEmb = WhisperOps.EnsureF32(w["audio_decoder.codebook_embeddings.weight"]);
        _fastEmb = WhisperOps.EnsureF32(w["audio_decoder.embeddings.weight"]);
        _fastOut = w["audio_decoder.output.weight"];
    }

    private static Dictionary<string, Tensor> RemapLayers(IReadOnlyDictionary<string, Tensor> w, string src, string dst,
        FishAudioTransformerConfig c)
    {
        int headDim = c.HeadDim > 0 ? c.HeadDim : c.HiddenSize / c.NumAttentionHeads;
        int q = c.NumAttentionHeads * headDim, kv = c.NumKeyValueHeads * headDim;
        Dictionary<string, Tensor> o = new();
        for (int i = 0; i < c.NumHiddenLayers; i++)
        {
            string s = $"{src}.layers.{i}", d = $"{dst}.layers.{i}";
            Tensor wqkv = w[$"{s}.attention.wqkv.weight"];
            o[$"{d}.self_attn.q_proj.weight"] = SliceRows(wqkv, 0, q);
            o[$"{d}.self_attn.k_proj.weight"] = SliceRows(wqkv, q, kv);
            o[$"{d}.self_attn.v_proj.weight"] = SliceRows(wqkv, q + kv, kv);
            o[$"{d}.self_attn.o_proj.weight"] = w[$"{s}.attention.wo.weight"];
            if (c.QkNorm)
            {
                o[$"{d}.self_attn.q_norm.weight"] = w[$"{s}.attention.q_norm.weight"];
                o[$"{d}.self_attn.k_norm.weight"] = w[$"{s}.attention.k_norm.weight"];
            }
            o[$"{d}.mlp.gate_proj.weight"] = w[$"{s}.feed_forward.w1.weight"];
            o[$"{d}.mlp.up_proj.weight"] = w[$"{s}.feed_forward.w3.weight"];
            o[$"{d}.mlp.down_proj.weight"] = w[$"{s}.feed_forward.w2.weight"];
            o[$"{d}.input_layernorm.weight"] = w[$"{s}.attention_norm.weight"];
            o[$"{d}.post_attention_layernorm.weight"] = w[$"{s}.ffn_norm.weight"];
        }
        return o;
    }

    // Copies rows [start, start+count) of a 2-D weight in its stored dtype (BF16 stays BF16).
    private static Tensor SliceRows(Tensor src, int start, int count)
    {
        if (src.DType.IsQuantized) throw new NotSupportedException("Fused wqkv must be an unquantized tensor.");
        int cols = (int)src.Shape[1];
        Tensor o = new(new TensorShape(count, cols), src.DType);
        long rowBytes = (long)cols * src.DType.SizeInBytes;
        Buffer.MemoryCopy((byte*)src.DataPointer + start * rowBytes, (void*)o.DataPointer, count * rowBytes, count * rowBytes);
        return o;
    }

    public IKvCache CreateSlowCache(int maxSeqLen) => KvCaches.ForDecode(
        _cfg.Slow.NumHiddenLayers, _cfg.Slow.NumKeyValueHeads, HeadDimOf(_cfg.Slow), maxSeqLen);

    private static int HeadDimOf(FishAudioTransformerConfig c) => c.HeadDim > 0 ? c.HeadDim : c.HiddenSize / c.NumAttentionHeads;

    private bool IsSemantic(int id) => id >= _cfg.SemanticBeginId && id <= _cfg.SemanticEndId;

    /// <summary>Summed input embedding of every prompt/generated position, <c>[1, T, hidden]</c>.
    /// <paramref name="tokens"/> is row 0; <paramref name="codes"/>[t] (length <see cref="FishAudioS2Config.NumCodebooks"/>)
    /// is the codebook column at t, or null for positions without codes. Codebook embeddings count only where row 0 is a
    /// semantic token, and those positions are scaled by <c>1/√(numCodebooks+1)</c> (upstream
    /// <c>scale_codebook_embeddings</c>).</summary>
    public Tensor EmbedFrames(ReadOnlySpan<int> tokens, IReadOnlyList<int[]?> codes)
    {
        int h = HiddenSize, n = _cfg.NumCodebooks, t = tokens.Length;
        Tensor outT = new(new TensorShape(1, t, h), DType.F32);
        _slow.EmbedLookup(outT, tokens);
        float* op = (float*)outT.DataPointer;
        float* cb = (float*)_codebookEmb!.DataPointer;
        float scale = 1f / MathF.Sqrt(n + 1);
        for (int p = 0; p < t; p++)
        {
            if (!IsSemantic(tokens[p])) continue;
            float* row = op + (long)p * h;
            int[]? c = codes[p];
            if (c is not null)
                for (int i = 0; i < n && i < c.Length; i++)
                {
                    float* e = cb + (long)(c[i] + i * _cfg.CodebookSize) * h;
                    for (int k = 0; k < h; k++) row[k] += e[k];
                }
            for (int k = 0; k < h; k++) row[k] *= scale;
        }
        return outT;
    }

    /// <summary>Runs the slow stack over <paramref name="embeds"/> and returns the POST-final-norm hidden state of the
    /// last position as <c>[1,1,hidden]</c>.</summary>
    public Tensor ForwardHidden(IBackend backend, Tensor embeds, int t, int posStart, IKvCache cache)
    {
        int h = HiddenSize;
        Tensor hidden = _slow.ForwardEmbeds(backend, embeds, t, posStart, cache, applyFinalNorm: true);
        if (t == 1) return hidden;
        Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        Buffer.MemoryCopy((float*)hidden.DataPointer + (long)(t - 1) * h, (void*)last.DataPointer, h * 4, h * 4);
        hidden.Dispose();
        return last;
    }

    /// <summary>Port of <c>decode_one_token_ar</c>: slow head → constrained semantic token (Repetition Aware Sampling
    /// against <paramref name="recent"/>, the trailing generated main tokens) → fast depth AR over the codebooks.
    /// Returns the main token and its <see cref="FishAudioS2Config.NumCodebooks"/> codes (code 0 is derived from the
    /// main token; the fast model samples codes 1..N-1).</summary>
    public (int Token, int[] Codes) SampleFrame(IBackend backend, Tensor hidden, ref uint rng, IReadOnlyList<int>? recent)
    {
        int n = _cfg.NumCodebooks, fastDim = _cfg.Fast.HiddenSize;

        using (Tensor logits = _slow.ProjectLogits(backend, hidden, 1))
        {
            ReadOnlySpan<float> l = new((void*)logits.DataPointer, _cfg.Slow.VocabSize);
            int token = FishAudioSampling.Sample(l, _cfg.Temperature, _cfg.TopP, _cfg.TopK, ref rng, Allowed());
            if (recent is not null && IsSemantic(token) && recent.Contains(token))
                token = FishAudioSampling.Sample(l, _cfg.RasTemperature, _cfg.RasTopP, _cfg.TopK, ref rng, Allowed());

            int[] codes = new int[n];
            codes[0] = Math.Clamp(token - _cfg.SemanticBeginId, 0, _cfg.CodebookSize - 1);
            using IKvCache cache = KvCaches.ForDecode(_cfg.Fast.NumHiddenLayers, _cfg.Fast.NumKeyValueHeads,
                HeadDimOf(_cfg.Fast), n + 1);

            // Position 0 only primes the fast cache with the slow hidden state (its logits are discarded upstream).
            _fast.ForwardEmbeds(backend, hidden, 1, 0, cache, applyFinalNorm: true).Dispose();
            Tensor input = FastEmbed(codes[0], fastDim);
            for (int k = 1; k < n; k++)
            {
                using Tensor normed = _fast.ForwardEmbeds(backend, input, 1, k, cache, applyFinalNorm: true);
                input.Dispose();
                using Tensor cl = WhisperOps.ProjectLinear(backend, normed, _fastOut!, null, 1, 1, fastDim, _cfg.CodebookSize);
                codes[k] = FishAudioSampling.Sample(new ReadOnlySpan<float>((void*)cl.DataPointer, _cfg.CodebookSize),
                    _cfg.Temperature, _cfg.TopP, _cfg.TopK, ref rng);
                input = k < n - 1 ? FastEmbed(codes[k], fastDim) : new Tensor(new TensorShape(1, 1, fastDim), DType.F32);
            }
            input.Dispose();
            return (token, codes);
        }
    }

    /// <summary>Diagnostics: teacher-forced slow pass over a whole sequence; returns the token logits
    /// <c>[T, vocab]</c> and the post-norm hidden states <c>[T, hidden]</c>.</summary>
    public (float[] Logits, float[] Hidden) DebugSlow(IBackend backend, int[] tokens, IReadOnlyList<int[]?> codes)
    {
        int t = tokens.Length, h = HiddenSize;
        using Tensor embeds = EmbedFrames(tokens, codes);
        using IKvCache cache = CreateSlowCache(t + 1);
        using Tensor hidden = _slow.ForwardEmbeds(backend, embeds, t, 0, cache, applyFinalNorm: true);
        using Tensor logits = _slow.ProjectLogits(backend, hidden, t);
        return (new ReadOnlySpan<float>((void*)logits.DataPointer, t * _cfg.Slow.VocabSize).ToArray(),
            new ReadOnlySpan<float>((void*)hidden.DataPointer, t * h).ToArray());
    }

    /// <summary>Diagnostics: runs the fast stack with the slow hidden state at position 0, then the embeddings of
    /// <paramref name="previousCodes"/> (codes 0..N-2) at positions 1..N-1; returns the logits <c>[N-1, codebook]</c>
    /// that predict codes 1..N-1.</summary>
    public float[] DebugFastLogits(IBackend backend, float[] hidden, int[] previousCodes)
    {
        int n = _cfg.NumCodebooks, dim = _cfg.Fast.HiddenSize, v = _cfg.CodebookSize;
        using IKvCache cache = KvCaches.ForDecode(_cfg.Fast.NumHiddenLayers, _cfg.Fast.NumKeyValueHeads, HeadDimOf(_cfg.Fast), n + 1);
        using Tensor first = new(new TensorShape(1, 1, dim), DType.F32);
        hidden.AsSpan(0, dim).CopyTo(new Span<float>((void*)first.DataPointer, dim));
        _fast.ForwardEmbeds(backend, first, 1, 0, cache, applyFinalNorm: true).Dispose();
        float[] outLogits = new float[(n - 1) * v];
        for (int k = 1; k < n; k++)
        {
            using Tensor input = FastEmbed(previousCodes[k - 1], dim);
            using Tensor normed = _fast.ForwardEmbeds(backend, input, 1, k, cache, applyFinalNorm: true);
            using Tensor cl = WhisperOps.ProjectLinear(backend, normed, _fastOut!, null, 1, 1, dim, v);
            new ReadOnlySpan<float>((void*)cl.DataPointer, v).CopyTo(outLogits.AsSpan((k - 1) * v, v));
        }
        return outLogits;
    }

    private ReadOnlySpan<int> Allowed()
    {
        if (_allowed is null)
        {
            int count = _cfg.SemanticEndId - _cfg.SemanticBeginId + 1;
            int[] a = new int[count + 1];
            for (int i = 0; i < count; i++) a[i] = _cfg.SemanticBeginId + i;
            a[count] = _cfg.ImEndId;
            _allowed = a;
        }
        return _allowed;
    }

    private Tensor FastEmbed(int code, int dim)
    {
        Tensor t = new(new TensorShape(1, 1, dim), DType.F32);
        Buffer.MemoryCopy((float*)_fastEmb!.DataPointer + (long)code * dim, (void*)t.DataPointer, dim * 4, dim * 4);
        return t;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _codebookEmb, _fastEmb, _fastOut }) if (t is not null) yield return t;
        foreach (Tensor t in _slow.EnumerateWeights()) yield return t;
        foreach (Tensor t in _fast.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _slow.Dispose(); _fast.Dispose();
        GC.SuppressFinalize(this);
    }
}
