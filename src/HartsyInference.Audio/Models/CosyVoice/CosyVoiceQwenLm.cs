using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.CosyVoice;

/// <summary>CosyVoice 2 text→speech-token language model: a Qwen2.5-0.5B backbone driving a unified text+speech autoregressive sequence.</summary>
/// <remarks>Mirrors <c>cosyvoice/llm/llm.py:Qwen2LM</c> rather than the
/// research doc's simplified "single extended-vocab softmax":
/// <list type="bullet">
///   <item><b>Text</b> tokens embed through the Qwen backbone's <c>embed_tokens</c>.</item>
///   <item><b>Speech</b> tokens embed through a <i>separate</i> <c>speech_embedding</c> table and are
///         predicted by a <i>separate</i> <c>llm_decoder</c> Linear head (the Qwen <c>lm_head</c> is
///         unused).</item>
///   <item><b>Control</b> tokens (<c>sos_eos</c>, <c>task_id</c>) come from a 2-row <c>llm_embedding</c>
///         that brackets the text/speech boundary.</item>
/// </list>
///
/// <para>Sequence (zero-shot, non-streaming): <c>[sos] ++ emb(prompt_text ++ text) ++ [task] ++
/// emb_speech(prompt_speech_tokens)</c>, then autoregress speech tokens until the end-of-speech token
/// (<see cref="CosyVoiceConfig.SpeechTokenSize"/>) is emitted.</para>
///
/// <para><b>Checkpoint-validation pending:</b> exact state-dict key prefixes are the documented
/// FunAudioLLM layout (<c>llm.model.model.*</c> backbone + top-level <c>speech_embedding</c> /
/// <c>llm_decoder</c> / <c>llm_embedding</c>); <see cref="LoadWeights"/> parameterizes them so they can
/// be reconciled against the real <c>llm.pt</c> when it's downloaded.</para></remarks>
public sealed unsafe class CosyVoiceQwenLm : IDisposable
{
    private readonly CosyVoiceConfig _cfg;
    private readonly Qwen2Model _backbone;
    private readonly int _speechVocab;       // SpeechTokenSize + SpeechTokenExtra (decoder/embedding rows)
    private readonly int _eosSpeechToken;    // == SpeechTokenSize
    private int _disposed;

    private Tensor? _speechEmbedding;        // [speechVocab, H]
    private Tensor? _llmDecoderW;            // [speechVocab, H]
    private Tensor? _llmDecoderB;            // [speechVocab]
    private Tensor? _llmEmbedding;           // [2, H] — row 0 = sos_eos, row 1 = task_id

    public CosyVoiceConfig Config => _cfg;

    /// <summary>Optional layer-split placement for the Qwen2.5-0.5B backbone — pools VRAM across GPUs. Null = single-backend on the per-call backend. The speech-decoder head follows <see cref="LlmPlacement.LastBackend"/> when set, mirroring <see cref="HartsyInference.Audio.Models.Music.YueStage1Lm.Placement"/>.</summary>
    public LlmPlacement? Placement { get; set; }

    public CosyVoiceQwenLm(CosyVoiceConfig cfg)
    {
        _cfg = cfg;
        _backbone = new Qwen2Model(cfg.Llm);
        _speechVocab = cfg.SpeechTokenSize + cfg.SpeechTokenExtra;
        _eosSpeechToken = cfg.SpeechTokenSize;
    }

    /// <summary>Loads the Qwen backbone plus the three speech-side heads. Key prefixes default to the FunAudioLLM layout; override per a real checkpoint dump.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w,
        string backbonePrefix = "llm.model.model",
        string speechEmbeddingKey = "speech_embedding.weight",
        string llmDecoderWeightKey = "llm_decoder.weight",
        string llmDecoderBiasKey = "llm_decoder.bias",
        string llmEmbeddingKey = "llm_embedding.weight")
    {
        ThrowIfDisposed();
        _backbone.LoadWeights(w, backbonePrefix);
        _speechEmbedding = WhisperOps.EnsureF32(w[speechEmbeddingKey]);
        _llmDecoderW = WhisperOps.EnsureF32(w[llmDecoderWeightKey]);
        _llmDecoderB = w.TryGetValue(llmDecoderBiasKey, out Tensor? b) ? WhisperOps.EnsureF32(b) : null;
        _llmEmbedding = WhisperOps.EnsureF32(w[llmEmbeddingKey]);
    }

    /// <summary>Autoregressively generates the speech-token stream for <paramref name="textTokens"/> (zero-shot, non-streaming). <paramref name="promptTextTokens"/> + <paramref name="promptSpeechTokens"/> are the reference clip's transcript tokens + its S3 speech tokens (empty for preset-voice modes). Returns the emitted speech-token IDs (EOS excluded).</summary>
    /// <param name="maxTokens">Hard cap on generated speech tokens (default ~ 30 s at 25 Hz).</param>
    /// <param name="seed">Deterministic sampling seed.</param>
    public List<int> GenerateSpeechTokens(IBackend backend,
        ReadOnlySpan<int> textTokens,
        ReadOnlySpan<int> promptTextTokens,
        ReadOnlySpan<int> promptSpeechTokens,
        int maxTokens = 750,
        int seed = 0,
        Action<int>? onToken = null)
    {
        ThrowIfDisposed();
        if (_speechEmbedding is null) throw new InvalidOperationException("CosyVoiceQwenLm weights not loaded.");

        int h = _cfg.Llm.HiddenSize;
        int textLen = promptTextTokens.Length + textTokens.Length;

        // ── Build the prompt embedding sequence: [sos] + text + [task] + prompt_speech ──
        int promptLen = 1 + textLen + 1 + promptSpeechTokens.Length;
        Tensor promptEmbeds = new(new TensorShape(1, promptLen, h), DType.F32);
        int off = 0;
        WriteControlRow(promptEmbeds, off++, 0);                       // sos_eos
        off = WriteTextEmbeds(backend, promptEmbeds, off, promptTextTokens);
        off = WriteTextEmbeds(backend, promptEmbeds, off, textTokens);
        WriteControlRow(promptEmbeds, off++, 1);                       // task_id
        off = WriteSpeechEmbeds(promptEmbeds, off, promptSpeechTokens);

        // ── KV cache sized for prompt + generation ──
        int cacheCap = Math.Min(_cfg.Llm.MaxPositionEmbeddings, promptLen + maxTokens + 8);
        using IKvCache cache = _backbone.CreateDecodeCache(cacheCap);

        List<int> generated = new(Math.Min(maxTokens, 256));
        SpeechSampler sampler = new(_cfg.Sampling, _eosSpeechToken, seed);

        // The layer-split path pools the body across the placement's backends; the speech-decoder head runs on the
        // last stage's backend, which owns llm_decoder (EnumerateStageWeights' contract) — same shape as YueStage1Lm.
        LlmPlacement? placement = Placement is { IsSingle: false } ? Placement : null;
        IBackend headBackend = placement?.LastBackend ?? backend;
        Tensor RunBody(Tensor embeds, int t, int posStart, IKvCache kv) => placement is null
            ? _backbone.ForwardEmbeds(backend, embeds, batch: 1, t: t, posStart: posStart, kv)
            : _backbone.ForwardEmbedsStaged(placement, embeds, t, posStart, kv);

        // Prefill, then sample from the last position.
        Tensor hidden = RunBody(promptEmbeds, promptLen, 0, cache);
        promptEmbeds.Dispose();

        for (int step = 0; step < maxTokens; step++)
        {
            int t = (int)hidden.Shape[1];
            Tensor last = VibeVoiceOps.SliceLastFrame(hidden, h);
            hidden.Dispose();
            Tensor logits = WhisperOps.ProjectLinear(headBackend, last, _llmDecoderW!, _llmDecoderB, 1, 1, h, _speechVocab);
            last.Dispose();

            int next = sampler.Sample(logits, generated);
            logits.Dispose();
            if (next == _eosSpeechToken) break;
            generated.Add(next);
            onToken?.Invoke(next);

            // Embed the new speech token and decode one step.
            Tensor stepEmbed = new(new TensorShape(1, 1, h), DType.F32);
            WriteSpeechRow(stepEmbed, 0, next);
            hidden = RunBody(stepEmbed, 1, cache.CurrentLength, cache);
            stepEmbed.Dispose();

            if (cache.CurrentLength >= cacheCap - 2) break;
        }
        hidden.Dispose();
        return generated;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor x in _backbone.EnumerateWeights()) yield return x;
        if (_speechEmbedding is not null) yield return _speechEmbedding;
        if (_llmDecoderW is not null) yield return _llmDecoderW;
        if (_llmDecoderB is not null) yield return _llmDecoderB;
        if (_llmEmbedding is not null) yield return _llmEmbedding;
    }

    /// <summary>One placement stage's weight tensors (asymmetric preload for the layer-split path): the backbone's layer range, plus, on the last stage, the speech-decoder head that <see cref="GenerateSpeechTokens"/> projects on <see cref="LlmPlacement.LastBackend"/>. <see cref="_speechEmbedding"/>/<see cref="_llmEmbedding"/> are read host-side only (raw <see cref="Tensor.DataPointer"/> gathers in <see cref="WriteSpeechRow"/>/<see cref="WriteControlRow"/>, never routed through <see cref="IBackend"/>) so they are excluded from every stage's preload set.</summary>
    public IEnumerable<Tensor> EnumerateStageWeights(int startLayer, int endLayer, bool isFirstStage, bool isLastStage,
        bool includeRedundantSplits = true)
    {
        foreach (Tensor t in _backbone.EnumerateStageWeights(startLayer, endLayer, isFirstStage, isLastStage, includeRedundantSplits))
            yield return t;
        if (isLastStage)
        {
            if (_llmDecoderW is not null) yield return _llmDecoderW;
            if (_llmDecoderB is not null) yield return _llmDecoderB;
        }
    }

    // ── Embedding writers (segment → rows of a [1, promptLen, H] buffer) ──

    private int WriteTextEmbeds(IBackend backend, Tensor dst, int rowOffset, ReadOnlySpan<int> ids)
    {
        if (ids.Length == 0) return rowOffset;
        int h = _cfg.Llm.HiddenSize;
        Tensor seg = new(new TensorShape(1, ids.Length, h), DType.F32);
        _backbone.EmbedLookup(seg, ids, batch: 1, t: ids.Length);
        CopyRows(seg, 0, dst, rowOffset, ids.Length, h);
        seg.Dispose();
        return rowOffset + ids.Length;
    }

    private int WriteSpeechEmbeds(Tensor dst, int rowOffset, ReadOnlySpan<int> ids)
    {
        for (int i = 0; i < ids.Length; i++) WriteSpeechRow(dst, rowOffset + i, ids[i]);
        return rowOffset + ids.Length;
    }

    private void WriteSpeechRow(Tensor dst, int row, int speechToken)
    {
        if ((uint)speechToken >= (uint)_speechVocab)
            throw new ArgumentException($"speech token {speechToken} out of range [0, {_speechVocab}).");
        int h = _cfg.Llm.HiddenSize;
        float* sp = (float*)_speechEmbedding!.DataPointer + (long)speechToken * h;
        float* dp = (float*)dst.DataPointer + (long)row * h;
        Buffer.MemoryCopy(sp, dp, h * 4, h * 4);
    }

    private void WriteControlRow(Tensor dst, int row, int controlIdx)
    {
        int h = _cfg.Llm.HiddenSize;
        float* sp = (float*)_llmEmbedding!.DataPointer + (long)controlIdx * h;
        float* dp = (float*)dst.DataPointer + (long)row * h;
        Buffer.MemoryCopy(sp, dp, h * 4, h * 4);
    }

    private static void CopyRows(Tensor src, int srcRow, Tensor dst, int dstRow, int count, int h)
    {
        float* sp = (float*)src.DataPointer + (long)srcRow * h;
        float* dp = (float*)dst.DataPointer + (long)dstRow * h;
        Buffer.MemoryCopy(sp, dp, (long)count * h * 4, (long)count * h * 4);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _backbone.Dispose();
        _speechEmbedding = null;
        _llmDecoderW = null;
        _llmDecoderB = null;
        _llmEmbedding = null;
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(CosyVoiceQwenLm));
    }
}
