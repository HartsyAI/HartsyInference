using SharpInference.Audio.Models.LanguageModels.Qwen2;
using SharpInference.Audio.Models.Whisper;
using SharpInference.Audio.Streaming;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.CosyVoice;

/// <summary>CosyVoice 2 text→speech-token language model: a Qwen2.5-0.5B backbone driving a unified
/// text+speech autoregressive sequence. Mirrors <c>cosyvoice/llm/llm.py:Qwen2LM</c> rather than the
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
/// be reconciled against the real <c>llm.pt</c> when it's downloaded.</para></summary>
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

    public CosyVoiceQwenLm(CosyVoiceConfig cfg)
    {
        _cfg = cfg;
        _backbone = new Qwen2Model(cfg.Llm);
        _speechVocab = cfg.SpeechTokenSize + cfg.SpeechTokenExtra;
        _eosSpeechToken = cfg.SpeechTokenSize;
    }

    /// <summary>Loads the Qwen backbone plus the three speech-side heads. Key prefixes default to the
    /// FunAudioLLM layout; override per a real checkpoint dump.</summary>
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

    /// <summary>Autoregressively generates the speech-token stream for <paramref name="textTokens"/>
    /// (zero-shot, non-streaming). <paramref name="promptTextTokens"/> + <paramref name="promptSpeechTokens"/>
    /// are the reference clip's transcript tokens + its S3 speech tokens (empty for preset-voice modes).
    /// Returns the emitted speech-token IDs (EOS excluded).</summary>
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
        using StreamingKvCache cache = new(_cfg.Llm.NumHiddenLayers, batch: 1,
            _cfg.Llm.NumKeyValueHeads, cacheCap, _cfg.Llm.HeadDim);

        List<int> generated = new(Math.Min(maxTokens, 256));
        SpeechSampler sampler = new(_cfg.Sampling, _eosSpeechToken, seed);

        // Prefill, then sample from the last position.
        Tensor hidden = _backbone.ForwardEmbeds(backend, promptEmbeds, batch: 1, t: promptLen, posStart: 0, cache);
        promptEmbeds.Dispose();

        for (int step = 0; step < maxTokens; step++)
        {
            int t = (int)hidden.Shape[1];
            Tensor last = SliceLastFrame(hidden, h);
            hidden.Dispose();
            Tensor logits = WhisperOps.ProjectLinear(backend, last, _llmDecoderW!, _llmDecoderB, 1, 1, h, _speechVocab);
            last.Dispose();

            int next = sampler.Sample(logits, generated);
            logits.Dispose();
            if (next == _eosSpeechToken) break;
            generated.Add(next);
            onToken?.Invoke(next);

            // Embed the new speech token and decode one step.
            Tensor stepEmbed = new(new TensorShape(1, 1, h), DType.F32);
            WriteSpeechRow(stepEmbed, 0, next);
            hidden = _backbone.ForwardEmbeds(backend, stepEmbed, batch: 1, t: 1, posStart: cache.CurrentLength, cache);
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

    private static Tensor SliceLastFrame(Tensor hidden, int h)
    {
        int t = (int)hidden.Shape[1];
        Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        float* sp = (float*)hidden.DataPointer + (long)(t - 1) * h;
        Buffer.MemoryCopy(sp, (void*)last.DataPointer, h * 4, h * 4);
        return last;
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
