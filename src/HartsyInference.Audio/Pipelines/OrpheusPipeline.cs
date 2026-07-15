using System.Diagnostics;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Codecs.Snac;
using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Audio.Models.Orpheus;
using HartsyInference.Audio.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using SnacModel = HartsyInference.Audio.Models.Codecs.Snac.Snac;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Orpheus TTS pipeline: a Llama-3.2-3B causal LM (<see cref="Qwen2Model"/>) autoregressively emits
/// a flat stream of SNAC audio tokens, which are redistributed into 3 hierarchical codebooks and decoded by
/// the built <see cref="SnacModel">SNAC</see> 24 kHz codec. Takes pre-tokenized text ids (the Audio package
/// carries no text-BPE dependency); the pipeline adds the model's human-frame control tokens. Reuses
/// <see cref="NucleusSampler"/> with repetition penalty applied by pre-shaping the logit buffer.</summary>
public sealed unsafe class OrpheusPipeline : IDisposable
{
    private readonly OrpheusConfig _cfg;
    private readonly Qwen2Model _backbone;
    private readonly SnacModel _codec;
    private int _disposed;

    public OrpheusPipeline(OrpheusConfig cfg)
    {
        _cfg = cfg;
        _backbone = new Qwen2Model(cfg.Llm);
        _codec = new SnacModel(cfg.Codec);
    }

    /// <summary>Sample rate of the emitted audio (24 kHz for SNAC 24 kHz).</summary>
    public int SampleRate => _cfg.Codec.SampleRate;

    /// <summary>Loads the Llama backbone (HF <c>LlamaForCausalLM</c> layout — <c>model.*</c> + tied head)
    /// and the SNAC codec weights.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> backbone, IReadOnlyDictionary<string, Tensor> snac,
        string backbonePrefix = "model")
    {
        _backbone.LoadWeights(backbone, backbonePrefix);
        _codec.LoadWeights(snac);
    }

    /// <summary>Synthesizes 24 kHz mono PCM from pre-tokenized text ids (e.g. the Llama BPE of
    /// <c>"{voice}: {text}"</c>). The pipeline wraps them in the <c>[StartOfHuman] … [EndOfText, EndOfHuman]</c>
    /// frame, AR-generates audio tokens until <see cref="OrpheusConfig.EndOfSpeech"/>, then SNAC-decodes.</summary>
    public float[] Synthesize(IBackend backend, ReadOnlySpan<int> textTokenIds, int maxTokens = 1200,
        int seed = 0, Action<GenerationProgress>? progress = null)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();

        // Orpheus prompt frame (matches the reference orpheus_tts._format_prompt): the caller's text ids already
        // carry the BOS token; wrap them as
        //   [StartOfHuman] textTokens [EndOfText, EndOfHuman, StartOfAi, StartOfSpeech]
        // The trailing StartOfAi + StartOfSpeech are what tell the model to begin emitting the audio-code stream —
        // without them the model produces unconditioned/gibberish speech (verified vs the transformers reference).
        int[] prompt = new int[textTokenIds.Length + 5];
        prompt[0] = _cfg.StartOfHuman;
        textTokenIds.CopyTo(prompt.AsSpan(1));
        prompt[^4] = _cfg.EndOfText;
        prompt[^3] = _cfg.EndOfHuman;
        prompt[^2] = _cfg.StartOfAi;
        prompt[^1] = _cfg.CodeStart;   // start-of-speech

        int cacheCap = Math.Min(_cfg.Llm.MaxPositionEmbeddings, prompt.Length + maxTokens + 8);
        using IKvCache cache = _backbone.CreateDecodeCache(cacheCap);

        uint rng = DeterministicRng.Seed(seed);
        int vocab = _cfg.Llm.VocabSize;
        float penalty = _cfg.RepetitionPenalty;
        HashSet<int> seen = new();
        List<int> generated = new(maxTokens);

        // The only tokens Orpheus validly emits are EndOfSpeech (128258) and the SNAC audio codes
        // (AudioCodeBase 128266 …), all ≥ CodeStart. Restricting the sampler to [CodeStart, vocab) skips the
        // ~128k orthographic-text logits (a small cut on the per-token host softmax+argsort). No quality change:
        // the excluded ids are never part of a valid speech continuation.
        // NOTE (perf pass 2026-07-14): the decode is GPU-compute-bound on the F32 backbone (~245 ms/token on a
        // 3060), not host- or launch-bound — sampler restriction and a CUDA-graph decode both measured flat. The
        // real lever is an F16 compute path in the shared GenericTransformer (benefits all LLM-backed audio
        // models); scoped as a separate cross-cutting project. See docs/Checklists/AUDIO_TTS_BRINGUP_PLAN.md.
        int sampleStart = _cfg.CodeStart;
        int sampleCount = vocab - sampleStart;

        bool prof = Environment.GetEnvironmentVariable("HARTSY_ORPHEUS_PROF") == "1";
        double tLogits = 0, tSample = 0, tFwd = 0; int profSteps = 0;
        Stopwatch psw = new();

        Tensor hidden = _backbone.Forward(backend, prompt, batch: 1, posStart: 0, cache);
        for (int step = 0; step < maxTokens; step++)
        {
            Tensor last = SliceLastFrame(hidden, _cfg.Llm.HiddenSize);
            hidden.Dispose();
            if (prof) { backend.Sync(); psw.Restart(); }
            Tensor logitsT = _backbone.ProjectLogits(backend, last, batch: 1, t: 1);
            if (prof) { backend.Sync(); tLogits += psw.Elapsed.TotalMilliseconds; psw.Restart(); }
            last.Dispose();

            Span<float> logits = new((void*)logitsT.DataPointer, vocab);
            if (penalty != 1f)
                foreach (int tok in seen)
                    logits[tok] = logits[tok] > 0 ? logits[tok] / penalty : logits[tok] * penalty;

            int next = sampleStart + NucleusSampler.Draw(logits.Slice(sampleStart, sampleCount), sampleCount,
                _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref rng);
            logitsT.Dispose();
            if (prof) { tSample += psw.Elapsed.TotalMilliseconds; }

            if (next == _cfg.EndOfSpeech) break;
            generated.Add(next);
            seen.Add(next);
            if (progress != null && (step & 63) == 0) progress(new GenerationProgress(step, maxTokens, sw.Elapsed.TotalMilliseconds));

            if (prof) { backend.Sync(); psw.Restart(); }
            hidden = _backbone.Forward(backend, [next], batch: 1, posStart: cache.CurrentLength, cache);
            if (prof) { backend.Sync(); tFwd += psw.Elapsed.TotalMilliseconds; profSteps++; }
            if (cache.CurrentLength >= cacheCap - 2) break;
        }
        if (prof && profSteps > 0)
            Logs.Info($"[Orpheus prof] {profSteps} steps: backbone-fwd {tFwd / profSteps:0.0}ms/tok, lm_head {tLogits / profSteps:0.0}ms/tok, sample {tSample / profSteps:0.0}ms/tok");

        int[] codes = OrpheusCodeFrames.ExtractAudioCodes(generated, _cfg.CodeStart, _cfg.EndOfSpeech, _cfg.TokensPerFrame);
        int groups = codes.Length / _cfg.TokensPerFrame;
        if (groups == 0)
        {
            Logs.Warning("Orpheus: no audio codes generated.");
            return [];
        }

        (int[] s1, int[] s2, int[] s3) = OrpheusCodeFrames.Redistribute(codes, _cfg.AudioCodeBase, _cfg.CodebookSize);
        Tensor l1 = ToCodeTensor(s1);
        Tensor l2 = ToCodeTensor(s2);
        Tensor l3 = ToCodeTensor(s3);
        Tensor audioT = _codec.Decode(backend, [l1, l2, l3], batch: 1);
        l1.Dispose(); l2.Dispose(); l3.Dispose();

        int n = (int)audioT.Shape[audioT.Shape.Rank - 1];
        float[] audio = new float[n];
        Buffer.MemoryCopy((void*)audioT.DataPointer,
            System.Runtime.CompilerServices.Unsafe.AsPointer(ref audio[0]), n * 4, n * 4);
        audioT.Dispose();
        sw.Stop();
        Logs.Info($"Orpheus: {groups} SNAC frames → {audio.Length} samples ({audio.Length / (double)SampleRate:F1}s) in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    private static Tensor ToCodeTensor(int[] codes)
    {
        Tensor t = new(new TensorShape(1, codes.Length), DType.I32);
        int* p = (int*)t.DataPointer;
        for (int i = 0; i < codes.Length; i++) p[i] = codes[i];
        return t;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
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
        _backbone.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(OrpheusPipeline));
    }
}
