using SharpInference.Audio.Dsp;
using SharpInference.Audio.Models.LanguageModels.Qwen2;
using SharpInference.Audio.Sampling;
using SharpInference.Audio.Streaming;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.SparkTts;

/// <summary>Spark-TTS text→BiCodec-token language model. A plain Qwen2.5-0.5B (extended vocab 166,000)
/// that autoregressively emits the unified token sequence — 32 global tokens then the 50 Hz semantic
/// stream — through the standard tied <c>lm_head</c>. Reuses <see cref="Qwen2Model"/> verbatim (no
/// separate speech heads, unlike CosyVoice) and the shared <see cref="NucleusSampler"/>; the only
/// Spark-specific logic is the chat-prompt-fed AR loop + parsing emitted absolute token IDs back into
/// global / semantic codec indices by ID range.</summary>
public sealed unsafe class SparkTtsLm : IDisposable
{
    private readonly SparkTtsConfig _cfg;
    private readonly Qwen2Model _backbone;
    private int _disposed;

    public SparkTtsConfig Config => _cfg;

    public SparkTtsLm(SparkTtsConfig cfg)
    {
        _cfg = cfg;
        _backbone = new Qwen2Model(cfg.Llm);
    }

    /// <summary>Loads the Qwen2.5 backbone (HF <c>Qwen2ForCausalLM</c> layout — <c>model.*</c> + tied
    /// <c>lm_head</c>). The extended 166,000-row embed/head load at their checkpoint shape.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string backbonePrefix = "model")
        => _backbone.LoadWeights(w, backbonePrefix);

    /// <summary>Runs the AR loop over <paramref name="promptTokenIds"/> (the chat-templated text + control
    /// prompt, built by the caller's Qwen tokenizer) and returns the emitted global + semantic codec
    /// indices (absolute LM token IDs mapped back through the config bases).</summary>
    /// <param name="maxTokens">Hard cap on generated tokens (~ 50 Hz × seconds).</param>
    public (List<int> Global, List<int> Semantic) GenerateAudioTokens(IBackend backend,
        ReadOnlySpan<int> promptTokenIds, int maxTokens = 1500, int seed = 0)
    {
        ThrowIfDisposed();
        int promptLen = promptTokenIds.Length;
        int cacheCap = Math.Min(_cfg.Llm.MaxPositionEmbeddings, promptLen + maxTokens + 8);
        using StreamingKvCache cache = new(_cfg.Llm.NumHiddenLayers, batch: 1,
            _cfg.Llm.NumKeyValueHeads, cacheCap, _cfg.Llm.HeadDim);

        uint rng = DeterministicRng.Seed(seed);
        List<int> global = new(_cfg.NumGlobalTokens);
        List<int> semantic = new(256);

        int vocab = _cfg.Llm.VocabSize;
        int[] promptArr = promptTokenIds.ToArray();
        Tensor hidden = _backbone.Forward(backend, promptArr, batch: 1, posStart: 0, cache);

        for (int step = 0; step < maxTokens; step++)
        {
            int t = (int)hidden.Shape[1];
            Tensor last = SliceLastFrame(hidden, _cfg.Llm.HiddenSize);
            hidden.Dispose();
            Tensor logits = _backbone.ProjectLogits(backend, last, batch: 1, t: 1);
            last.Dispose();

            int next = NucleusSampler.Draw(
                new Span<float>((void*)logits.DataPointer, vocab), vocab,
                _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref rng);
            logits.Dispose();

            if (next == _cfg.EosTokenId) break;

            if (next >= _cfg.GlobalTokenBase && next < _cfg.GlobalTokenBase + _cfg.GlobalVocab)
                global.Add(next - _cfg.GlobalTokenBase);
            else if (next >= _cfg.SemanticTokenBase && next < _cfg.SemanticTokenBase + _cfg.SemanticVocab)
                semantic.Add(next - _cfg.SemanticTokenBase);
            // Structure/control tokens (start/end markers) are phase separators — not codec tokens.

            int[] step1 = [next];
            hidden = _backbone.Forward(backend, step1, batch: 1, posStart: cache.CurrentLength, cache);
            if (cache.CurrentLength >= cacheCap - 2) break;
        }
        hidden.Dispose();
        return (global, semantic);
    }

    public IEnumerable<Tensor> EnumerateWeights() => _backbone.EnumerateWeights();

    private Tensor SliceLastFrame(Tensor hidden, int h)
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
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(SparkTtsLm));
    }
}
