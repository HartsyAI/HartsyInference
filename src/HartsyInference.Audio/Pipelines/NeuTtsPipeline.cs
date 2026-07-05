using System.Diagnostics;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Codecs.NeuCodec;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.NeuTts;
using HartsyInference.Audio.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>NeuTTS Air pipeline: a Qwen2.5-0.5B LM emits a single NeuCodec FSQ stream which the
/// <see cref="NeuCodecDecoder"/> turns into 24 kHz audio. Voice cloning conditions on reference NeuCodec codes
/// spliced into the prompt. Takes the pre-tokenized prompt prefix + reference codes (caller phonemizes with
/// espeak and tokenizes via <see cref="NeuTtsPromptBuilder"/>; the Audio package carries no text-BPE
/// dependency). Reuses <see cref="Qwen2Model"/> + <see cref="NucleusSampler"/> (top-k draw).</summary>
public sealed unsafe class NeuTtsPipeline : IDisposable
{
    private readonly NeuTtsConfig _cfg;
    private readonly Qwen2Model _lm;
    private readonly NeuCodecDecoder _codec;
    private int _disposed;

    public NeuTtsPipeline(NeuTtsConfig cfg)
    {
        _cfg = cfg;
        _lm = new Qwen2Model(cfg.Llm);
        _codec = new NeuCodecDecoder(cfg.Codec);
    }

    public int SampleRate => _cfg.Codec.SampleRate;

    /// <summary>Loads the Qwen2.5 backbone (HF <c>Qwen2ForCausalLM</c> layout, tied head) and NeuCodec.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> backbone, IReadOnlyDictionary<string, Tensor> codec,
        string backbonePrefix = "model")
    {
        _lm.LoadWeights(backbone, backbonePrefix);
        _codec.LoadWeights(codec);
    }

    /// <summary>Synthesizes 24 kHz mono PCM. <paramref name="promptPrefix"/> is the tokenized chat prefix from
    /// <see cref="NeuTtsPromptBuilder.BuildPromptPrefix"/> (verified against upstream <c>neutts.py</c>
    /// <c>_apply_chat_template</c>); this appends <c>SPEECH_GENERATION_START</c> + <paramref name="refCodes"/>
    /// (reference NeuCodec FSQ indices) exactly as upstream. Upstream always clones from a reference — an empty
    /// span (default voice) is a best-effort extension, not an upstream mode. Stops on
    /// <c>SPEECH_GENERATION_END</c>; total length capped at <see cref="NeuTtsConfig.MaxContext"/> (upstream
    /// <c>max_length</c>); sampling temp 1.0 / top-k 50 / min-new-tokens 50 (upstream <c>_infer_torch</c>).</summary>
    public float[] Synthesize(IBackend backend, ReadOnlySpan<int> promptPrefix, ReadOnlySpan<int> refCodes,
        int maxTokens = 2000, int seed = 0, Action<GenerationProgress>? progress = null, int minNewTokens = -1)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();
        int minNew = minNewTokens >= 0 ? minNewTokens : _cfg.MinNewTokens;

        int[] prompt = new int[promptPrefix.Length + 1 + refCodes.Length];
        promptPrefix.CopyTo(prompt);
        prompt[promptPrefix.Length] = _cfg.SpeechGenStart;
        for (int i = 0; i < refCodes.Length; i++)
            prompt[promptPrefix.Length + 1 + i] = _cfg.SpeechTokenBase + refCodes[i];

        int cacheCap = Math.Min(Math.Min(_cfg.Llm.MaxPositionEmbeddings, _cfg.MaxContext),
            prompt.Length + maxTokens + 8);
        using IKvCache cache = _lm.CreateDecodeCache(cacheCap);

        uint rng = DeterministicRng.Seed(seed);
        int vocab = _cfg.Llm.VocabSize;
        List<int> codes = new(maxTokens);

        Span<int> recentBuf = _cfg.RepetitionPenalty > 1f ? stackalloc int[_cfg.RepetitionWindow] : default;
        Tensor hidden = _lm.Forward(backend, prompt, batch: 1, posStart: 0, cache);
        for (int step = 0; step < maxTokens; step++)
        {
            Tensor last = SliceLastFrame(hidden, _cfg.Llm.HiddenSize);
            hidden.Dispose();
            Tensor logitsT = _lm.ProjectLogits(backend, last, batch: 1, t: 1);
            last.Dispose();

            Span<float> logits = new((void*)logitsT.DataPointer, vocab);
            if (codes.Count < minNew) logits[_cfg.SpeechGenEnd] = float.NegativeInfinity;
            if (_cfg.RepetitionPenalty > 1f && codes.Count > 0)
            {
                int win = Math.Min(_cfg.RepetitionWindow, codes.Count);
                Span<int> recent = recentBuf[..win];
                for (int r = 0; r < win; r++) recent[r] = _cfg.SpeechTokenBase + codes[codes.Count - win + r];
                RepetitionPenalty.Apply(logits, recent, _cfg.RepetitionPenalty);
            }
            int next = NucleusSampler.Draw(logits, vocab, _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref rng);
            logitsT.Dispose();

            if (next == _cfg.SpeechGenEnd) break;
            int code = next - _cfg.SpeechTokenBase;
            if ((uint)code < (uint)_cfg.CodebookSize) codes.Add(code);
            if (progress != null && (step & 63) == 0) progress(new(step, maxTokens, sw.Elapsed.TotalMilliseconds));

            int[] one = [next];
            hidden = _lm.Forward(backend, one, batch: 1, posStart: cache.CurrentLength, cache);
            if (cache.CurrentLength >= cacheCap - 2) break;
        }
        hidden.Dispose();

        if (codes.Count == 0) { Logs.Warning("NeuTTS: no speech codes generated."); return []; }
        float[] audio = _codec.Decode(backend, codes.ToArray());
        sw.Stop();
        Logs.Info($"NeuTTS: {codes.Count} codes → {audio.Length} samples ({audio.Length / (double)SampleRate:F1}s) in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _lm.EnumerateWeights()) yield return t;
        foreach (Tensor t in _codec.EnumerateWeights()) yield return t;
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
        _lm.Dispose();
        _codec.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(NeuTtsPipeline));
    }
}
