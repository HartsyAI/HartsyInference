using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.LanguageModels.Gpt;

/// <summary>GPT-2-style pre-norm Transformer backbone (multi-head attention, 4× GELU MLP, LayerNorm) shared by the package's GPT-2-family audio LMs — Bark's three bias-free stages, IndexTTS's biased standard HF GPT-2 — parameterized by depth / width / head count rather than copied per model. Position handling is the caller's choice: a single shared learned table (<see cref="LoadWeights"/>'s <c>posKey</c>) or externally pre-applied embeddings (<c>positionsApplied</c>, for checkpoints with per-segment or no backbone-owned position table).</summary>
/// <remarks>Operates on caller-supplied input embeddings <c>[1, T, hidden]</c> (each model owns its own token / codebook embedding tables and output heads). Supports causal and non-causal (full bidirectional, e.g. Bark-Fine). AR decoding is incremental: one cache-capturing prefill <see cref="Forward"/> then O(T) per-token <see cref="ForwardStep"/> calls against a device-resident <see cref="IKvCache"/> — projections, attention (FlashAttention), and the K/V cache all stay on the backend, so nothing crosses back to the host mid-forward.</remarks>
public sealed unsafe class GptBackbone : IDisposable
{
    private readonly GptConfig _cfg;
    private readonly GptBlock[] _blocks;
    private int _disposed;

    private Tensor? _posEmbed;     // [blockSize, hidden] learned absolute positions
    private Tensor? _lnFGamma, _lnFBeta;

    public GptConfig Config => _cfg;

    public GptBackbone(GptConfig cfg)
    {
        _cfg = cfg;
        _blocks = new GptBlock[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++) _blocks[i] = new GptBlock(cfg);
    }

    /// <summary>Keys follow the HF Bark scheme by default (<c>position_embeds_layer.weight</c>, <c>layers.{i}.*</c>, <c>layernorm_final.*</c>); pass the model's own keys for other checkpoints. <paramref name="posKey"/> is null for a checkpoint with no backbone-owned position table at all (e.g. IndexTTS, whose HF GPT-2 export has its <c>wpe</c> stripped because the caller always supplies pre-positioned embeddings via <c>positionsApplied: true</c>) — every <see cref="Forward"/>/<see cref="ForwardStep"/> call must then pass that flag, since <see cref="AddPositions"/> would dereference a null table otherwise.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string? posKey, string blockPrefix,
        string lnFGammaKey, string lnFBetaKey)
    {
        _posEmbed = posKey is null ? null : WhisperOps.EnsureF32(w[posKey]);
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"{blockPrefix}.{i}");
        _lnFGamma = WhisperOps.EnsureF32(w[lnFGammaKey]);
        _lnFBeta = GptBlock.LoadBiasOrZero(w, lnFBetaKey, _lnFGamma);
    }

    /// <summary>Runs the stack over <paramref name="inputEmbeds"/> <c>[1, T, hidden]</c>, positioned from the cache's committed length (0 if none); supplying <paramref name="cache"/> makes this the incremental-decode PREFILL (every layer's K/V appended into the device cache, enabling continuation via <see cref="ForwardStep"/>), otherwise it's a plain full-sequence forward (causal unless <paramref name="nonCausal"/>, used by the bidirectional Bark-Fine stage).</summary>
    /// <param name="positionsApplied">True when the caller already added its own positional embedding(s) to <paramref name="inputEmbeds"/> — e.g. IndexTTS, whose text and mel segments each restart from position 0 under separate learned tables rather than one table spanning the whole sequence. Skips this backbone's own single shared <see cref="_posEmbed"/> addition.</param>
    /// <returns><c>[1, T, hidden]</c>.</returns>
    public Tensor Forward(IBackend backend, Tensor inputEmbeds, bool nonCausal = false, IKvCache? cache = null, bool positionsApplied = false)
    {
        ThrowIfDisposed();
        int t = (int)inputEmbeds.Shape[1];
        int h = _cfg.Hidden;
        int posStart = cache?.CurrentLength ?? 0;
        if (cache is not null && posStart + t > _cfg.BlockSize)
        {
            throw new ArgumentException($"Prefill length {t} at offset {posStart} exceeds block size {_cfg.BlockSize}.");
        }

        Tensor hidden = positionsApplied ? CloneF32(inputEmbeds) : AddPositions(inputEmbeds, t, h, posStart);

        if (cache is not null)
        {
            for (int i = 0; i < _blocks.Length; i++)
            {
                Tensor next = _blocks[i].ForwardCached(backend, hidden, cache, i, posStart);
                hidden.Dispose();
                hidden = next;
            }
            cache.AdvanceLength(t);
        }
        else
        {
            Tensor? causalMask = (!nonCausal && t > 1) ? BuildCausalMask(t) : null;
            for (int i = 0; i < _blocks.Length; i++)
            {
                Tensor next = _blocks[i].Forward(backend, hidden, causalMask);
                hidden.Dispose();
                hidden = next;
            }
            causalMask?.Dispose();
        }

        Tensor normed = new(hidden.Shape, DType.F32);
        backend.LayerNorm(normed, hidden, _lnFGamma!, _lnFBeta!, 1e-5f);
        hidden.Dispose();
        return normed;
    }

    /// <summary>Creates an empty device-resident K/V cache sized to the model's block size; Bark is MHA, so the K/V head count equals the query head count.</summary>
    public IKvCache CreateCache() => KvCaches.ForDecode(_cfg.NumLayers, _cfg.NumHeads, _cfg.HeadDim, _cfg.BlockSize);

    /// <summary>Runs one token's embedding <c>[1, 1, hidden]</c> through the stack against <paramref name="cache"/> (appending its K/V at <see cref="IKvCache.CurrentLength"/>) — equivalent to the last position of a full-sequence causal <see cref="Forward"/> over the cached prefix plus this token, but at O(T) instead of O(T²).</summary>
    /// <param name="positionsApplied">See <see cref="Forward"/>; when true the caller has already added the right positional embedding for this token (e.g. a segment-relative one) to <paramref name="inputEmbed"/>.</param>
    public Tensor ForwardStep(IBackend backend, Tensor inputEmbed, IKvCache cache, bool positionsApplied = false)
    {
        ThrowIfDisposed();
        int pos = cache.CurrentLength;
        if (pos >= _cfg.BlockSize)
        {
            throw new InvalidOperationException($"KV cache is full ({_cfg.BlockSize}); the caller must stop at the model's block size.");
        }
        int h = _cfg.Hidden;

        Tensor hidden = positionsApplied ? CloneF32(inputEmbed) : AddPositions(inputEmbed, 1, h, pos);
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].ForwardCached(backend, hidden, cache, i, pos);
            hidden.Dispose();
            hidden = next;
        }
        cache.AdvanceLength(1);

        Tensor normed = new(hidden.Shape, DType.F32);
        backend.LayerNorm(normed, hidden, _lnFGamma!, _lnFBeta!, 1e-5f);
        hidden.Dispose();
        return normed;
    }

    /// <summary>Copies <paramref name="inputEmbeds"/> into a fresh buffer the block loop owns and disposes per-layer, for the <c>positionsApplied</c> path where no position embedding is added here.</summary>
    private static Tensor CloneF32(Tensor inputEmbeds)
    {
        Tensor copy = new(inputEmbeds.Shape, DType.F32);
        float* ip = (float*)inputEmbeds.DataPointer;
        float* op = (float*)copy.DataPointer;
        for (long i = 0; i < copy.ElementCount; i++) op[i] = ip[i];
        return copy;
    }

    /// <summary>Adds the learned position embedding for <c>[posStart, posStart+t)</c>; runs host-side since the input is a freshly host-built embedding lookup that hasn't touched the device yet.</summary>
    private Tensor AddPositions(Tensor inputEmbeds, int t, int h, int posStart)
    {
        Tensor hidden = new(inputEmbeds.Shape, DType.F32);
        float* ip = (float*)inputEmbeds.DataPointer;
        float* op = (float*)hidden.DataPointer;
        float* pe = (float*)_posEmbed!.DataPointer + (long)posStart * h;
        for (int s = 0; s < t; s++)
        {
            long off = (long)s * h;
            for (int c = 0; c < h; c++) op[off + c] = ip[off + c] + pe[off + c];
        }
        return hidden;
    }

    private static Tensor BuildCausalMask(int t)
    {
        Tensor mask = new(new TensorShape(1, 1, t, t), DType.F32);
        float* mp = (float*)mask.DataPointer;
        for (int q = 0; q < t; q++)
            for (int k = 0; k < t; k++)
                mp[(long)q * t + k] = k <= q ? 0f : float.NegativeInfinity;
        return mask;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_posEmbed is not null) yield return _posEmbed;
        foreach (GptBlock b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
        if (_lnFGamma is not null) yield return _lnFGamma;
        if (_lnFBeta is not null) yield return _lnFBeta;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (GptBlock b in _blocks) b.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(GptBackbone));
    }
}
