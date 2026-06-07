using SharpInference.Audio.Models.Whisper;
using SharpInference.Audio.Streaming;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.LanguageModels.Qwen2;

/// <summary>Full Qwen2 / Qwen2.5 autoregressive language model: embedding lookup + stack
/// of <see cref="Qwen2DecoderLayer"/>s + final RMSNorm + optional <c>lm_head</c>. Designed
/// for the two-phase generation pattern:
/// <list type="number">
///   <item><b>Prefill</b> — call <see cref="Forward"/> with all prompt tokens in one shot
///         (T &gt; 1). The causal mask blocks attention to future positions. Cache is
///         populated for positions <c>[0, T)</c>.</item>
///   <item><b>Decode</b> — call <see cref="Forward"/> with a single new token. The cache
///         already covers all earlier positions; the new token attends to all of them
///         (no mask needed because there's only one query position).</item>
/// </list>
///
/// <para>Supports the VibeVoice <b>inputs_embeds</b> path too: when the caller passes
/// <see cref="ForwardEmbeds"/> directly with already-computed embeddings (e.g. for the
/// speech-input-mask insertion the multi-speaker prefill does), the embedding lookup is
/// skipped.</para>
///
/// <para>The streaming-0.5B variant of VibeVoice splits the LM into a 4-layer text encoder
/// and a 20-layer TTS upper. <see cref="ForwardEmbeds"/> takes optional <c>startLayer</c>
/// / <c>endLayer</c> parameters so the caller can run an arbitrary contiguous slice of
/// the layers and feed the output back later.</para></summary>
public sealed unsafe class Qwen2Model : IDisposable
{
    private readonly Qwen2Config _cfg;
    private readonly Qwen2RotaryEmbedding _rope;
    private readonly Qwen2DecoderLayer[] _layers;
    private Tensor? _embed;
    private Tensor? _finalNorm;
    private Tensor? _lmHead;     // null when tied — we re-use _embed.
    private int _disposed;

    public Qwen2Config Config => _cfg;
    public int NumLayers => _cfg.NumHiddenLayers;
    public int HiddenSize => _cfg.HiddenSize;
    public int NumKeyValueHeads => _cfg.NumKeyValueHeads;
    public int HeadDim => _cfg.HeadDim;

    public Qwen2Model(Qwen2Config cfg)
    {
        _cfg = cfg;
        _rope = new Qwen2RotaryEmbedding(cfg.HeadDim, cfg.RopeTheta, cfg.MaxPositionEmbeddings);
        _layers = new Qwen2DecoderLayer[cfg.NumHiddenLayers];
        for (int i = 0; i < cfg.NumHiddenLayers; i++) _layers[i] = new Qwen2DecoderLayer(cfg, _rope, i);
    }

    /// <summary>Loads all LM weights from a HF-style key dict. <paramref name="prefix"/>
    /// is everything up to (but not including) <c>embed_tokens</c> — for VibeVoice that's
    /// <c>"model.language_model"</c>, for a standalone Qwen2 checkpoint it's <c>"model"</c>.
    ///
    /// <para>If <see cref="Qwen2Config.TieWordEmbeddings"/> is true, <c>lm_head</c> is the
    /// same tensor as <c>embed_tokens</c>. Otherwise an explicit <c>lm_head.weight</c> is
    /// loaded — for VibeVoice-Large that key lives at <c>lm_head.weight</c> (no
    /// <c>model.</c> prefix).</para></summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, string? lmHeadKey = null)
    {
        ThrowIfDisposed();
        _embed = WhisperOps.EnsureF32(w[$"{prefix}.embed_tokens.weight"]);
        _finalNorm = WhisperOps.EnsureF32(w[$"{prefix}.norm.weight"]);
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, $"{prefix}.layers.{i}");

        if (!_cfg.TieWordEmbeddings)
        {
            string key = lmHeadKey ?? "lm_head.weight";
            _lmHead = WhisperOps.EnsureF32(w[key]);
        }
    }

    /// <summary>Loads a <b>headless</b> transformer body — the decoder layers + final RMSNorm only, no
    /// <c>embed_tokens</c> / <c>lm_head</c>. Used when the embedding tables and output heads live on an
    /// outer model (e.g. Sesame CSM's backbone/decoder, whose <c>tok_embeddings</c>/<c>output</c> are
    /// <c>Identity</c>). Drive it via <see cref="ForwardEmbeds"/>; do not call <see cref="ProjectLogits"/>.</summary>
    public void LoadWeightsHeadless(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        ThrowIfDisposed();
        _finalNorm = WhisperOps.EnsureF32(w[$"{prefix}.norm.weight"]);
        for (int i = 0; i < _layers.Length; i++) _layers[i].LoadWeights(w, $"{prefix}.layers.{i}");
    }

    /// <summary>Standard token-IDs-in path. Performs embedding lookup then delegates to
    /// <see cref="ForwardEmbeds"/>.
    ///
    /// <para>Returns the final <c>[B, T, hidden]</c> hidden state (already passed through
    /// the final RMSNorm). Apply <see cref="ProjectLogits"/> to get next-token logits.</para></summary>
    public Tensor Forward(IBackend backend, ReadOnlySpan<int> tokenIds, int batch, int posStart, StreamingKvCache cache)
    {
        ThrowIfDisposed();
        int t = tokenIds.Length / Math.Max(1, batch);
        if (batch * t != tokenIds.Length)
            throw new ArgumentException($"token count ({tokenIds.Length}) must equal batch ({batch}) × T.");

        Tensor embeds = new(new TensorShape(batch, t, _cfg.HiddenSize), DType.F32);
        EmbedLookup(embeds, tokenIds, batch, t);
        Tensor output = ForwardEmbeds(backend, embeds, batch, t, posStart, cache);
        embeds.Dispose();
        return output;
    }

    /// <summary>Embedding-in path — used by VibeVoice when the caller has already replaced
    /// speech-mask positions with their connector outputs.
    ///
    /// <para><paramref name="startLayer"/> and <paramref name="endLayer"/> control which
    /// contiguous slice of decoder layers runs. Default is the full stack
    /// <c>[0, NumLayers)</c>. Streaming-0.5B uses <c>(0, 4)</c> for the lower text encoder
    /// (replaces final norm with identity by passing <c>applyFinalNorm: false</c>) and
    /// <c>(4, 24)</c> for the upper TTS backbone (final norm + lm_head as usual).</para></summary>
    public Tensor ForwardEmbeds(IBackend backend, Tensor inputsEmbeds, int batch, int t, int posStart,
        StreamingKvCache cache, int startLayer = 0, int? endLayer = null, bool applyFinalNorm = true)
    {
        ThrowIfDisposed();
        int last = endLayer ?? _layers.Length;
        if (startLayer < 0 || last > _layers.Length || startLayer > last)
            throw new ArgumentException($"Invalid layer range [{startLayer}, {last}) for a stack of {_layers.Length}.");

        // Build a causal mask if this is a multi-token forward (prefill). For single-token
        // decode (t=1) no mask is needed — the single query naturally attends to all
        // cached keys, which is exactly what causal attention wants.
        Tensor? causalMask = t > 1 ? BuildCausalMask(t, posStart, cache.CurrentLength + t) : null;
        try
        {
            Tensor hidden = inputsEmbeds;
            bool ownsHidden = false;
            for (int i = startLayer; i < last; i++)
            {
                Tensor next = _layers[i].Forward(backend, hidden, batch, t, posStart, cache, causalMask);
                if (ownsHidden) hidden.Dispose();
                hidden = next;
                ownsHidden = true;
            }

            // Advance the cache once per step, after all layers have appended.
            cache.AdvanceLength(t);

            if (applyFinalNorm)
            {
                Tensor normed = new(hidden.Shape, DType.F32);
                backend.RmsNorm(normed, hidden, _finalNorm!, _cfg.RmsNormEps);
                if (ownsHidden) hidden.Dispose();
                return normed;
            }
            // Return the raw last-layer hidden state — caller may want pre-norm features.
            if (!ownsHidden)
            {
                // Clone so the caller doesn't accidentally share the input buffer.
                Tensor copy = new(hidden.Shape, DType.F32);
                Buffer.MemoryCopy((void*)hidden.DataPointer, (void*)copy.DataPointer, hidden.ElementCount * 4, hidden.ElementCount * 4);
                return copy;
            }
            return hidden;
        }
        finally
        {
            causalMask?.Dispose();
        }
    }

    /// <summary>Projects the final hidden state <c>[B, T, hidden]</c> to vocab logits
    /// <c>[B, T, vocab]</c>. For tied-embedding models the embed_tokens table is used
    /// directly. The caller typically only cares about <c>logits[:, -1, :]</c> (the
    /// next-token distribution) — pass a single-token-T hidden if you only need that
    /// slice.</summary>
    public Tensor ProjectLogits(IBackend backend, Tensor hidden, int batch, int t)
    {
        ThrowIfDisposed();
        Tensor headW = _lmHead ?? _embed ?? throw new InvalidOperationException("Qwen2Model weights not loaded.");
        Tensor logits = WhisperOps.ProjectLinear(backend, hidden, headW, bias: null, batch, t, _cfg.HiddenSize, _cfg.VocabSize);
        return logits;
    }

    /// <summary>Performs the embedding lookup for a batch of token IDs.</summary>
    public void EmbedLookup(Tensor output, ReadOnlySpan<int> tokenIds, int batch, int t)
    {
        ThrowIfDisposed();
        int h = _cfg.HiddenSize;
        float* op = (float*)output.DataPointer;
        float* ep = (float*)_embed!.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < t; s++)
            {
                int id = tokenIds[b * t + s];
                if ((uint)id >= (uint)_cfg.VocabSize)
                    throw new ArgumentException($"token id {id} out of range [0, {_cfg.VocabSize}).");
                float* row = ep + (long)id * h;
                long outOff = ((long)b * t + s) * h;
                Buffer.MemoryCopy(row, op + outOff, h * 4, h * 4);
            }
        }
    }

    /// <summary>Builds a 4-D additive attention mask <c>[1, 1, T_q, T_k]</c> with
    /// lower-triangular zero and upper-triangular <c>-inf</c>. <paramref name="qOffset"/>
    /// is the absolute position of the first query row (the cache length before this
    /// step), so a query at row <c>q</c> can attend to keys at positions
    /// <c>0..qOffset+q</c> inclusive. SharpInference's SDPA accepts the
    /// <c>[1, 1, Sq, Skv]</c> shape and broadcasts across batch + head dims.</summary>
    private static Tensor BuildCausalMask(int tq, int qOffset, int tk)
    {
        Tensor mask = new(new TensorShape(1, 1, tq, tk), DType.F32);
        float* p = (float*)mask.DataPointer;
        const float NegInf = -1e30f;
        for (int q = 0; q < tq; q++)
        {
            int absQ = qOffset + q;
            for (int k = 0; k < tk; k++)
                p[q * tk + k] = k <= absQ ? 0f : NegInf;
        }
        return mask;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(Qwen2Model));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _embed = null;
        _finalNorm = null;
        _lmHead = null;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_embed is not null) yield return _embed;
        if (_finalNorm is not null) yield return _finalNorm;
        if (_lmHead is not null) yield return _lmHead;
        foreach (Qwen2DecoderLayer layer in _layers)
            foreach (Tensor t in layer.EnumerateWeights()) yield return t;
    }
}
