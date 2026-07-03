using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.FishSpeech;

/// <summary>Fish-Speech DualAR model: a slow backbone (<see cref="Qwen2Model"/>) consumes the summed
/// text + codebook embeddings and predicts the next semantic token; its hidden state then drives a small fast
/// (depth) <see cref="Qwen2Model"/> that autoregressively predicts the 8 audio codebooks for the frame. Codebook
/// embeddings share one offset table. Reuses <see cref="NucleusSampler"/>.</summary>
public sealed unsafe class FishSpeechDualAr : IDisposable
{
    private readonly FishSpeechConfig _cfg;
    private readonly Qwen2Model _backbone;
    private readonly Qwen2Model _fast;
    private Tensor? _textEmb, _codebookEmb, _slowHead, _fastEmb, _fastOut, _fastNorm, _fastProjIn, _slowNorm;
    private int _disposed;

    public FishSpeechDualAr(FishSpeechConfig cfg)
    {
        _cfg = cfg;
        _backbone = new Qwen2Model(cfg.Backbone);
        _fast = new Qwen2Model(cfg.Fast);
    }

    public int HiddenSize => _cfg.Backbone.HiddenSize;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        // The fish-speech checkpoint is a Llama variant with FUSED keys and no "model." prefix:
        // layers.{i}.attention.wqkv (fused q|k|v), .wo; feed_forward.w1/w3/w2 (gate/up/down); attention_norm /
        // ffn_norm; final norm. Remap to the split keys Qwen2Model expects (no biases — AttentionBias=false).
        _backbone.LoadWeightsHeadless(Remap(w, "layers", _cfg.Backbone, "model"), "model");
        _fast.LoadWeightsHeadless(Remap(w, "fast_layers", _cfg.Fast, "fast_model"), "fast_model");

        _textEmb = WhisperOps.EnsureF32(w["embeddings.weight"]);
        _codebookEmb = WhisperOps.EnsureF32(w["codebook_embeddings.weight"]);
        _slowHead = WhisperOps.EnsureF32(w["output.weight"]);
        _fastEmb = WhisperOps.EnsureF32(w["fast_embeddings.weight"]);
        _fastOut = WhisperOps.EnsureF32(w["fast_output.weight"]);
        _fastNorm = WhisperOps.EnsureF32(w["fast_norm.weight"]);
        _slowNorm = WhisperOps.EnsureF32(w["norm.weight"]);   // slow final RMSNorm (for the slow head; the fast model takes the PRE-norm hidden)
        _fastProjIn = w.TryGetValue("fast_project_in.weight", out Tensor? p) ? WhisperOps.EnsureF32(p) : null;
    }

    // Converts the fish Llama keys (one of slow "layers" / fast "fast_layers") into the split keys Qwen2Model
    // wants under {dstPrefix}.*, slicing the fused wqkv into q/k/v by head counts.
    private static Dictionary<string, Tensor> Remap(IReadOnlyDictionary<string, Tensor> w, string src,
        Qwen2Config cfg, string dstPrefix)
    {
        int q = cfg.NumAttentionHeads * cfg.HeadDim, kv = cfg.NumKeyValueHeads * cfg.HeadDim;
        Dictionary<string, Tensor> outW = new();
        for (int i = 0; ; i++)
        {
            if (!w.TryGetValue($"{src}.{i}.attention.wqkv.weight", out Tensor? wqkv)) break;
            Tensor f = WhisperOps.EnsureF32(wqkv);
            int inDim = (int)f.Shape[1];
            string p = $"{dstPrefix}.layers.{i}";
            outW[$"{p}.self_attn.q_proj.weight"] = SliceRows(f, 0, q, inDim);
            outW[$"{p}.self_attn.k_proj.weight"] = SliceRows(f, q, kv, inDim);
            outW[$"{p}.self_attn.v_proj.weight"] = SliceRows(f, q + kv, kv, inDim);
            outW[$"{p}.self_attn.o_proj.weight"] = WhisperOps.EnsureF32(w[$"{src}.{i}.attention.wo.weight"]);
            outW[$"{p}.mlp.gate_proj.weight"] = WhisperOps.EnsureF32(w[$"{src}.{i}.feed_forward.w1.weight"]);
            outW[$"{p}.mlp.up_proj.weight"] = WhisperOps.EnsureF32(w[$"{src}.{i}.feed_forward.w3.weight"]);
            outW[$"{p}.mlp.down_proj.weight"] = WhisperOps.EnsureF32(w[$"{src}.{i}.feed_forward.w2.weight"]);
            outW[$"{p}.input_layernorm.weight"] = WhisperOps.EnsureF32(w[$"{src}.{i}.attention_norm.weight"]);
            outW[$"{p}.post_attention_layernorm.weight"] = WhisperOps.EnsureF32(w[$"{src}.{i}.ffn_norm.weight"]);
        }
        string finalNorm = src == "fast_layers" ? "fast_norm.weight" : "norm.weight";
        outW[$"{dstPrefix}.norm.weight"] = WhisperOps.EnsureF32(w[finalNorm]);
        return outW;
    }

    private static Tensor SliceRows(Tensor src, int rowStart, int rowCount, int cols)
    {
        Tensor o = new(new TensorShape(rowCount, cols), DType.F32);
        Buffer.MemoryCopy((float*)src.DataPointer + (long)rowStart * cols, (void*)o.DataPointer,
            (long)rowCount * cols * 4, (long)rowCount * cols * 4);
        return o;
    }

    /// <summary>True when <paramref name="id"/> is a <c>&lt;|semantic:i|&gt;</c> token (or the range is unset).</summary>
    private bool IsSemantic(int id) => _cfg.SemanticBeginId < 0
        || (id >= _cfg.SemanticBeginId && id < _cfg.SemanticBeginId + _cfg.CodebookSize);

    /// <summary>Builds the summed input embedding for one frame: <c>text_emb(row0) + Σ codebook_emb(code_i +
    /// i·codebook_size)</c>. Matches upstream <c>BaseTransformer.embed</c>: the codebook sum is ZEROED when row-0
    /// is not a <c>&lt;|semantic:i|&gt;</c> token (text prompt positions carry only the text embedding). Returns
    /// <c>[1,1,hidden]</c>.</summary>
    public Tensor EmbedFrame(int semantic, ReadOnlySpan<int> codes)
    {
        Tensor outT = new(new TensorShape(1, 1, HiddenSize), DType.F32);
        EmbedFrameInto((float*)outT.DataPointer, semantic, codes);
        return outT;
    }

    /// <summary>Builds the summed embeddings for a whole prompt (codebook rows all pad) as <c>[1,T,hidden]</c>,
    /// for one batched prefill forward.</summary>
    public Tensor EmbedPrompt(ReadOnlySpan<int> row0)
    {
        int h = HiddenSize;
        Tensor outT = new(new TensorShape(1, row0.Length, h), DType.F32);
        float* op = (float*)outT.DataPointer;
        Span<int> pad = stackalloc int[_cfg.NumCodebooks];
        for (int i = 0; i < row0.Length; i++) EmbedFrameInto(op + (long)i * h, row0[i], pad);
        return outT;
    }

    private void EmbedFrameInto(float* op, int semantic, ReadOnlySpan<int> codes)
    {
        int h = HiddenSize, n = _cfg.NumCodebooks;
        float* tp = (float*)_textEmb!.DataPointer + (long)semantic * h;
        float* cb = (float*)_codebookEmb!.DataPointer;
        for (int c = 0; c < h; c++) op[c] = tp[c];
        if (IsSemantic(semantic))
        {
            for (int i = 0; i < n && i < codes.Length; i++)
            {
                float* row = cb + (long)(codes[i] + i * _cfg.CodebookSize) * h;
                for (int c = 0; c < h; c++) op[c] += row[c];
            }
        }
        if (_cfg.ScaleCodebookEmbeddings)
        {
            float scale = 1f / MathF.Sqrt(n + 1);
            for (int c = 0; c < h; c++) op[c] *= scale;
        }
    }

    /// <summary>Allocates the slow-backbone decode cache (efficient <see cref="FixedKvCache"/>) for the AR loop;
    /// pass it to <see cref="GenerateFrame"/> each step. Dispose when the utterance completes.</summary>
    public IKvCache CreateSlowCache(int maxSeqLen) => _backbone.CreateDecodeCache(maxSeqLen);

    /// <summary>Runs the slow backbone over <paramref name="frameEmbeds"/> (<c>[1,T,hidden]</c>) and returns the
    /// PRE-final-norm hidden state of the LAST position as <c>[1,1,hidden]</c> (upstream <c>forward_generate</c>:
    /// the slow head normalizes, the fast model consumes the pre-norm hidden).</summary>
    public Tensor ForwardHidden(IBackend backend, Tensor frameEmbeds, int t, int posStart, IKvCache slowCache)
    {
        int h = HiddenSize;
        Tensor hidden = _backbone.ForwardEmbeds(backend, frameEmbeds, 1, t, posStart, slowCache, applyFinalNorm: false);
        if (t == 1) return hidden;
        Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        Buffer.MemoryCopy((float*)hidden.DataPointer + (long)(t - 1) * h, (void*)last.DataPointer, h * 4, h * 4);
        hidden.Dispose();
        return last;
    }

    /// <summary>Samples one frame from the pre-norm slow hidden (<c>[1,1,hidden]</c>), the port of upstream
    /// <c>decode_one_token_ar</c>: slow head → semantic token (full text vocab); codebook 0 is DERIVED as
    /// <c>semantic − SemanticBeginId</c> (clamped ≥ 0, pos-0 fast logits discarded); the fast depth AR then
    /// samples codebooks 1..N−1. <paramref name="recent"/> = trailing generated frames (each
    /// <c>[semantic, code0..codeN−1]</c>) for the windowed repetition penalty; null = no penalty.</summary>
    public (int Semantic, int[] Codes) SampleFrame(IBackend backend, Tensor hidden, ref uint rng,
        IReadOnlyList<int[]>? recent = null)
    {
        int h = HiddenSize, n = _cfg.NumCodebooks;
        Tensor slowPost = new(hidden.Shape, DType.F32);
        backend.RmsNorm(slowPost, hidden, _slowNorm!, _cfg.Backbone.RmsNormEps);

        // Slow head → semantic token.
        Tensor slowLogits = WhisperOps.ProjectLinear(backend, slowPost, _slowHead!, null, 1, 1, h, _cfg.TextVocab);
        slowPost.Dispose();
        Span<float> sl = new((void*)slowLogits.DataPointer, _cfg.TextVocab);
        ApplyRepetitionPenalty(sl, recent, 0);
        int semantic = NucleusSampler.Draw(sl, _cfg.TextVocab, _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref rng);
        slowLogits.Dispose();

        // Fast depth transformer over the codebook axis (fresh cache per frame, as upstream).
        int[] codes = new int[n];
        int fastDim = _cfg.Fast.HiddenSize;
        using IKvCache fastCache = _fast.CreateDecodeCache(n + 1);
        Tensor depthIn = ProjectSlow(backend, hidden, fastDim);   // step 0 input = slow hidden
        int start = 0;
        if (_cfg.SemanticBeginId >= 0)
        {
            // Upstream: pos-0 fast forward only fills the cache; codebook 0 comes from the semantic token.
            _fast.ForwardEmbeds(backend, depthIn, 1, 1, 0, fastCache, applyFinalNorm: false).Dispose();
            depthIn.Dispose();
            int a = semantic - _cfg.SemanticBeginId;
            codes[0] = Math.Clamp(a, 0, _cfg.CodebookSize - 1);
            depthIn = EmbedFast(codes[0], fastDim);
            start = 1;
        }
        for (int k = start; k < n; k++)
        {
            Tensor fh = _fast.ForwardEmbeds(backend, depthIn, 1, 1, k, fastCache, applyFinalNorm: false);
            depthIn.Dispose();
            Tensor normed = new(fh.Shape, DType.F32);
            backend.RmsNorm(normed, fh, _fastNorm!, _cfg.Fast.RmsNormEps); fh.Dispose();
            Tensor cl = WhisperOps.ProjectLinear(backend, normed, _fastOut!, null, 1, 1, fastDim, _cfg.CodebookSize);
            normed.Dispose();
            Span<float> fl = new((void*)cl.DataPointer, _cfg.CodebookSize);
            ApplyRepetitionPenalty(fl, recent, k + 1);
            int tok = NucleusSampler.Draw(fl, _cfg.CodebookSize, _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref rng);
            cl.Dispose();
            codes[k] = tok;
            if (k < n - 1) depthIn = EmbedFast(tok, fastDim);
            else depthIn = new Tensor(new TensorShape(1, 1, fastDim), DType.F32);   // unused last
        }
        depthIn.Dispose();
        return (semantic, codes);
    }

    /// <summary>One frame: slow step → next semantic token + the N codebook tokens (fast depth AR).</summary>
    public (int Semantic, int[] Codes) GenerateFrame(IBackend backend, Tensor frameEmbed, int posStart,
        IKvCache slowCache, ref uint rng)
    {
        Tensor hidden = ForwardHidden(backend, frameEmbed, 1, posStart, slowCache);
        (int Semantic, int[] Codes) r = SampleFrame(backend, hidden, ref rng);
        hidden.Dispose();
        return r;
    }

    // Upstream logits_to_probs: penalize each distinct token in the trailing window once (score<0 ? ×p : ÷p).
    private void ApplyRepetitionPenalty(Span<float> logits, IReadOnlyList<int[]>? recent, int row)
    {
        float p = _cfg.RepetitionPenalty;
        if (recent is null || recent.Count == 0 || p == 1f) return;
        HashSet<int> seen = [];
        foreach (int[] frame in recent)
        {
            int tok = frame[row];
            if ((uint)tok >= (uint)logits.Length || !seen.Add(tok)) continue;
            float s = logits[tok];
            logits[tok] = s < 0 ? s * p : s / p;
        }
    }

    /// <summary>Diagnostics — runs the full-sequence slow path (summed embed → backbone → slow head) and returns
    /// the token logits <c>[1, T, vocab]</c> for parity checks. <paramref name="codes"/> is per-frame codebook ids.</summary>
    public Tensor DebugSlowLogits(IBackend backend, int[] semantics, int[][] codes)
    {
        int t = semantics.Length, h = HiddenSize;
        Tensor embed = new(new TensorShape(1, t, h), DType.F32);
        float* ep = (float*)embed.DataPointer;
        for (int i = 0; i < t; i++)
        {
            using Tensor fe = EmbedFrame(semantics[i], codes[i]);
            Buffer.MemoryCopy((void*)fe.DataPointer, ep + (long)i * h, h * 4, h * 4);
        }
        using IKvCache cache = _backbone.CreateDecodeCache(t + 1);
        Tensor hidden = _backbone.ForwardEmbeds(backend, embed, 1, t, 0, cache);
        embed.Dispose();
        Tensor logits = WhisperOps.ProjectLinear(backend, hidden, _slowHead!, null, 1, t, h, _cfg.TextVocab);
        hidden.Dispose();
        return logits;
    }

    /// <summary>Diagnostics — runs the fast depth transformer over the full codebook axis (slow hidden at
    /// position 0 + the first N-1 codebook embeddings) and returns the per-position codebook logits
    /// <c>[1, N, codebook_size]</c>. <paramref name="hidden"/> is the slow PRE-norm hidden.</summary>
    public Tensor DebugFastLogits(IBackend backend, float[] hidden, int[] prevCodes)
    {
        int dim = _cfg.Fast.HiddenSize, n = _cfg.NumCodebooks;
        Tensor x = new(new TensorShape(1, n, dim), DType.F32);
        float* xp = (float*)x.DataPointer;
        for (int c = 0; c < dim; c++) xp[c] = hidden[c];               // pos 0 = slow hidden (fast_project_in = identity)
        float* fe = (float*)_fastEmb!.DataPointer;
        for (int i = 0; i < n - 1; i++)
            for (int c = 0; c < dim; c++) xp[(long)(i + 1) * dim + c] = fe[(long)prevCodes[i] * dim + c];

        using IKvCache cache = _fast.CreateDecodeCache(n + 1);
        Tensor hid = _fast.ForwardEmbeds(backend, x, 1, n, 0, cache, applyFinalNorm: false);
        x.Dispose();
        Tensor normed = new(hid.Shape, DType.F32);
        backend.RmsNorm(normed, hid, _fastNorm!, _cfg.Fast.RmsNormEps);
        hid.Dispose();
        Tensor logits = WhisperOps.ProjectLinear(backend, normed, _fastOut!, null, 1, n, dim, _cfg.CodebookSize);
        normed.Dispose();
        return logits;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_textEmb, _codebookEmb, _slowHead, _fastEmb, _fastOut, _fastNorm, _fastProjIn, _slowNorm];
        foreach (Tensor? t in own) if (t is not null) yield return t;
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
        foreach (Tensor t in _fast.EnumerateWeights()) yield return t;
    }

    private Tensor ProjectSlow(IBackend backend, Tensor hidden, int fastDim)
    {
        if (_fastProjIn is null)
        {
            Tensor copy = new(new TensorShape(1, 1, fastDim), DType.F32);
            Buffer.MemoryCopy((void*)hidden.DataPointer, (void*)copy.DataPointer, fastDim * 4, fastDim * 4);
            return copy;
        }
        return WhisperOps.ProjectLinear(backend, hidden, _fastProjIn, null, 1, 1, HiddenSize, fastDim);
    }

    private Tensor EmbedFast(int token, int fastDim)
    {
        Tensor t = new(new TensorShape(1, 1, fastDim), DType.F32);
        Buffer.MemoryCopy((float*)_fastEmb!.DataPointer + (long)token * fastDim, (void*)t.DataPointer, fastDim * 4, fastDim * 4);
        return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _backbone.Dispose(); _fast.Dispose();
        GC.SuppressFinalize(this);
    }
}
