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
    private static readonly bool _dbg = Environment.GetEnvironmentVariable("HARTSY_ORPHEUS_DEBUG") == "1";
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

        int[] prompt = new int[textTokenIds.Length + 3];
        prompt[0] = _cfg.StartOfHuman;
        textTokenIds.CopyTo(prompt.AsSpan(1));
        prompt[^2] = _cfg.EndOfText;
        prompt[^1] = _cfg.EndOfHuman;

        int cacheCap = Math.Min(_cfg.Llm.MaxPositionEmbeddings, prompt.Length + maxTokens + 8);
        using IKvCache cache = _backbone.CreateDecodeCache(cacheCap);

        uint rng = DeterministicRng.Seed(seed);
        int vocab = _cfg.Llm.VocabSize;
        float penalty = _cfg.RepetitionPenalty;
        HashSet<int> seen = new();
        List<int> generated = new(maxTokens);

        Tensor hidden = _backbone.Forward(backend, prompt, batch: 1, posStart: 0, cache);
        for (int step = 0; step < maxTokens; step++)
        {
            Tensor last = SliceLastFrame(hidden, _cfg.Llm.HiddenSize);
            hidden.Dispose();
            Tensor logitsT = _backbone.ProjectLogits(backend, last, batch: 1, t: 1);
            last.Dispose();

            Span<float> logits = new((void*)logitsT.DataPointer, vocab);
            if (penalty != 1f)
                foreach (int tok in seen)
                    logits[tok] = logits[tok] > 0 ? logits[tok] / penalty : logits[tok] * penalty;

            int next = NucleusSampler.Draw(logits, vocab, _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref rng);
            logitsT.Dispose();

            if (next == _cfg.EndOfSpeech) { if (_dbg) Logs.Info($"[Orpheus dbg] STOP EndOfSpeech at step {step}, generated={generated.Count}"); break; }
            generated.Add(next);
            seen.Add(next);
            if (progress != null && (step & 63) == 0) progress(new GenerationProgress(step, maxTokens, sw.Elapsed.TotalMilliseconds));

            int[] step1 = [next];
            hidden = _backbone.Forward(backend, step1, batch: 1, posStart: cache.CurrentLength, cache);
            if (cache.CurrentLength >= cacheCap - 2) break;
        }
        hidden.Dispose();

        if (_dbg)
        {
            int inRange = generated.Count(g => g >= _cfg.AudioCodeBase && g < _cfg.AudioCodeBase + 7 * _cfg.CodebookSize);
            Logs.Info($"[Orpheus dbg] total generated={generated.Count}, in-audio-range={inRange}, AudioCodeBase={_cfg.AudioCodeBase}, EndOfSpeech={_cfg.EndOfSpeech}");
            Logs.Info($"[Orpheus dbg] first 40 tokens: {string.Join(",", generated.Take(40))}");
            Logs.Info($"[Orpheus dbg] prompt frame: SOH={_cfg.StartOfHuman} EOT={_cfg.EndOfText} EOH={_cfg.EndOfHuman}, textTokens={textTokenIds.Length}, first text ids: {string.Join(",", textTokenIds.ToArray().Take(12))}");
        }
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
