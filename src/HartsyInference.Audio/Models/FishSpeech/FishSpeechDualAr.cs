using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Sampling;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

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
    private Tensor? _textEmb, _codebookEmb, _slowHead, _fastEmb, _fastOut, _fastNorm, _fastProjIn;
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

    /// <summary>Builds the summed input embedding for one frame: <c>text_emb(sem) + Σ codebook_emb(code_i +
    /// i·codebook_size)</c>, scaled by <c>1/√(N+1)</c>. Returns <c>[1,1,hidden]</c>.</summary>
    public Tensor EmbedFrame(int semantic, ReadOnlySpan<int> codes)
    {
        int h = HiddenSize, n = _cfg.NumCodebooks;
        Tensor outT = new(new TensorShape(1, 1, h), DType.F32);
        float* op = (float*)outT.DataPointer;
        float* tp = (float*)_textEmb!.DataPointer + (long)semantic * h;
        float* cb = (float*)_codebookEmb!.DataPointer;
        for (int c = 0; c < h; c++) op[c] = tp[c];
        for (int i = 0; i < n && i < codes.Length; i++)
        {
            float* row = cb + (long)(codes[i] + i * _cfg.CodebookSize) * h;
            for (int c = 0; c < h; c++) op[c] += row[c];
        }
        if (_cfg.ScaleCodebookEmbeddings)
        {
            float scale = 1f / MathF.Sqrt(n + 1);
            for (int c = 0; c < h; c++) op[c] *= scale;
        }
        return outT;
    }

    /// <summary>One frame: slow step → next semantic token + the 8 codebook tokens (fast depth AR).</summary>
    public (int Semantic, int[] Codes) GenerateFrame(IBackend backend, Tensor frameEmbed, int posStart,
        StreamingKvCache slowCache, ref uint rng)
    {
        int h = HiddenSize, n = _cfg.NumCodebooks;
        Tensor hidden = _backbone.ForwardEmbeds(backend, frameEmbed, 1, 1, posStart, slowCache);

        // Slow head → semantic token.
        Tensor slowLogits = WhisperOps.ProjectLinear(backend, hidden, _slowHead!, null, 1, 1, h, _cfg.TextVocab);
        int semantic = NucleusSampler.Draw(new Span<float>((void*)slowLogits.DataPointer, _cfg.TextVocab),
            _cfg.TextVocab, _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref rng);
        slowLogits.Dispose();

        // Fast depth transformer over the codebook axis.
        int[] codes = new int[n];
        int fastDim = _cfg.Fast.HiddenSize;
        using StreamingKvCache fastCache = new(_cfg.Fast.NumHiddenLayers, 1, _cfg.Fast.NumKeyValueHeads, n + 1, _cfg.Fast.HeadDim);
        Tensor depthIn = ProjectSlow(backend, hidden, fastDim);   // step 0 input = slow hidden
        hidden.Dispose();
        for (int k = 0; k < n; k++)
        {
            Tensor fh = _fast.ForwardEmbeds(backend, depthIn, 1, 1, k, fastCache);
            depthIn.Dispose();
            Tensor normed = new(fh.Shape, DType.F32);
            backend.RmsNorm(normed, fh, _fastNorm!, _cfg.Fast.RmsNormEps); fh.Dispose();
            Tensor cl = WhisperOps.ProjectLinear(backend, normed, _fastOut!, null, 1, 1, fastDim, _cfg.CodebookSize);
            normed.Dispose();
            int tok = NucleusSampler.Draw(new Span<float>((void*)cl.DataPointer, _cfg.CodebookSize),
                _cfg.CodebookSize, _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref rng);
            cl.Dispose();
            codes[k] = tok;
            if (k < n - 1) depthIn = EmbedFast(tok, fastDim);
            else depthIn = new Tensor(new TensorShape(1, 1, fastDim), DType.F32);   // unused last
        }
        depthIn.Dispose();
        return (semantic, codes);
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
        using StreamingKvCache cache = new(_cfg.Backbone.NumHiddenLayers, 1, _cfg.Backbone.NumKeyValueHeads, t + 1, _cfg.Backbone.HeadDim);
        Tensor hidden = _backbone.ForwardEmbeds(backend, embed, 1, t, 0, cache);
        embed.Dispose();
        Tensor logits = WhisperOps.ProjectLinear(backend, hidden, _slowHead!, null, 1, t, h, _cfg.TextVocab);
        hidden.Dispose();
        return logits;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] own = [_textEmb, _codebookEmb, _slowHead, _fastEmb, _fastOut, _fastNorm, _fastProjIn];
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
