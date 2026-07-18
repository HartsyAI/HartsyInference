using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Kyutai;

/// <summary>Kyutai STT model: a headless Helium backbone (<see cref="Qwen2Model"/>) plus the single shared
/// <c>embed_tokens</c> table that holds both text rows and the 32 audio-codebook ranges. Per frame the input
/// embedding is the sum of the previous text token's row and the 32 audio codes' rows (each offset into its
/// codebook's range); the tied head projects the final hidden over the text-vocab rows only.</summary>
public sealed unsafe class KyutaiSttModel : IDisposable
{
    private readonly KyutaiSttConfig _cfg;
    private readonly Qwen2Model _backbone;
    private Tensor? _embed;
    private Tensor? _head;    // separate text output head (lm_head.weight); null → tied to _embed
    private int _disposed;

    public KyutaiSttConfig Config => _cfg;
    public int HiddenSize => _cfg.Helium.HiddenSize;

    public KyutaiSttModel(KyutaiSttConfig cfg)
    {
        _cfg = cfg;
        _backbone = new Qwen2Model(cfg.Helium);
    }

    /// <summary>Loads the shared embedding table (<c>model.embed_tokens.embed_tokens.weight</c>) and the
    /// headless Helium body (layers + final norm), reconciling the <c>-trfs</c> checkpoint layout first.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "model",
        string embedKey = "model.embed_tokens.embed_tokens.weight")
    {
        _embed = WhisperOps.EnsureF32(w[embedKey]);
        // The -trfs checkpoint carries a SEPARATE text head (lm_head.weight [TextVocab, hidden]); use it when
        // present — the shared embed_tokens table is NOT tied to the output projection here. Fall back to the
        // tied embed rows for checkpoints that omit lm_head.
        _head = w.TryGetValue("lm_head.weight", out Tensor? h) ? WhisperOps.EnsureF32(h) : null;
        _backbone.LoadWeightsHeadless(ReconcileHeliumLayout(w, prefix), prefix);
    }

    /// <summary>The kyutai <c>-trfs</c> (HF Transformers) checkpoint ships the Helium body in a layout the shared
    /// <see cref="Qwen2Model"/>/GenericTransformer loader doesn't read directly: each attention projection is
    /// wrapped as <c>{q,k,v,o}_proj.linear.weight</c> (an extra <c>.linear</c> from the quant-aware Linear
    /// module), and the gated MLP is a single fused <c>mlp.fc1.weight</c> <c>[2·inter, hidden]</c> (gate ‖ up,
    /// in that order — moshi's <c>linear_in</c> then <c>view(…,2,-1)</c>) plus <c>mlp.fc2.weight</c> (down).
    /// Rewrite those to the standard <c>{q,k,v,o}_proj.weight</c> / <c>gate_proj|up_proj|down_proj.weight</c>
    /// keys. A checkpoint that lacks the <c>.linear</c> marker passes through unchanged, so a future
    /// moshi-native load still works.</summary>
    private static IReadOnlyDictionary<string, Tensor> ReconcileHeliumLayout(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        if (!w.ContainsKey($"{prefix}.layers.0.self_attn.q_proj.linear.weight")) return w;   // not the -trfs layout
        Dictionary<string, Tensor> outW = new(w.Count);
        foreach (KeyValuePair<string, Tensor> kv in w)
        {
            string k = kv.Key;
            if (k.EndsWith(".linear.weight", StringComparison.Ordinal) && k.Contains(".self_attn.", StringComparison.Ordinal))
            {
                outW[string.Concat(k.AsSpan(0, k.Length - ".linear.weight".Length), ".weight")] = kv.Value;
            }
            else if (k.EndsWith(".mlp.fc1.weight", StringComparison.Ordinal))
            {
                string basep = k[..^".fc1.weight".Length];
                (Tensor gate, Tensor up) = SplitFusedRows(kv.Value);
                outW[basep + ".gate_proj.weight"] = gate;
                outW[basep + ".up_proj.weight"] = up;
            }
            else if (k.EndsWith(".mlp.fc2.weight", StringComparison.Ordinal))
            {
                outW[k[..^".fc2.weight".Length] + ".down_proj.weight"] = kv.Value;
            }
            else
            {
                outW[k] = kv.Value;
            }
        }
        return outW;
    }

    /// <summary>Splits a fused <c>[2·rows, cols]</c> weight into its two contiguous row-halves (dtype-preserving
    /// byte copy) — recovers <c>gate</c> (first half) and <c>up</c> (second half) from the checkpoint's fused
    /// <c>fc1</c>.</summary>
    private static (Tensor Gate, Tensor Up) SplitFusedRows(Tensor fused)
    {
        int rows = (int)fused.Shape[0], cols = (int)fused.Shape[1];
        int half = rows / 2;
        long halfBytes = fused.DType.ComputeByteCount((long)half * cols);
        Tensor gate = new(new TensorShape(half, cols), fused.DType);
        Tensor up = new(new TensorShape(half, cols), fused.DType);
        byte* src = (byte*)fused.DataPointer;
        Buffer.MemoryCopy(src, (void*)gate.DataPointer, halfBytes, halfBytes);
        Buffer.MemoryCopy(src + halfBytes, (void*)up.DataPointer, halfBytes, halfBytes);
        return (gate, up);
    }

    /// <summary>Builds the per-frame input embedding <c>[1,1,hidden]</c> = text-row + Σ audio-code rows.</summary>
    public Tensor EmbedFrame(int prevTextToken, ReadOnlySpan<int> audioCodes)
    {
        if (_embed is null) throw new InvalidOperationException("KyutaiSttModel weights not loaded.");
        int h = HiddenSize;
        Tensor outT = new(new TensorShape(1, 1, h), DType.F32);
        float* op = (float*)outT.DataPointer;
        float* ep = (float*)_embed.DataPointer;
        AddRow(op, ep, prevTextToken, h);
        for (int k = 0; k < _cfg.NumCodebooks && k < audioCodes.Length; k++)
            AddRow(op, ep, _cfg.AudioOffset(k) + audioCodes[k], h);
        return outT;
    }

    /// <summary>Runs one Helium step over a prebuilt frame embedding and returns the final hidden state.</summary>
    public Tensor Step(IBackend backend, Tensor frameEmbed, int posStart, StreamingKvCache cache)
        => _backbone.ForwardEmbeds(backend, frameEmbed, batch: 1, t: 1, posStart, cache);

    /// <summary>Projects a final hidden state to text logits over the first <c>TextVocab</c> rows of the
    /// shared (tied) embedding.</summary>
    public Tensor ProjectText(IBackend backend, Tensor hidden)
    {
        if (_embed is null) throw new InvalidOperationException("KyutaiSttModel weights not loaded.");
        return WhisperOps.ProjectLinear(backend, hidden, _head ?? _embed, null, 1, 1,
            _cfg.Helium.HiddenSize, _cfg.TextVocab);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embed is not null) yield return _embed;
        if (_head is not null) yield return _head;
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
    }

    private static void AddRow(float* dst, float* table, int row, int h)
    {
        float* src = table + (long)row * h;
        for (int i = 0; i < h; i++) dst[i] += src[i];
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _embed = null;
        _backbone.Dispose();
        GC.SuppressFinalize(this);
    }
}
