using HartsyInference.Audio.Models.Zonos;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.Audio.Phonemizer.Espeak;

namespace HartsyInference.Audio.Pipelines;

/// <summary>End-to-end Zonos-v0.1 voice-cloning TTS: text + a reference clip → 44.1 kHz mono. Owns and wires the
/// four Zonos components — the <see cref="ZonosSpeakerEncoder"/> (ref wav → 128-d speaker), the
/// <see cref="ZonosPhonemeTokenizer"/> (espeak → phoneme ids), the <see cref="ZonosConditioning"/> prefix builder
/// (cond + CFG-uncond), and the <see cref="ZonosPipeline"/> delayed-AR generator + DAC decode.</summary>
public sealed class ZonosTts : IDisposable
{
    private readonly ZonosConfig _cfg;
    private readonly ZonosPipeline _pipeline;
    private readonly ZonosConditioning _conditioning;
    private readonly ZonosSpeakerEncoder _speaker;
    private readonly ZonosPhonemeTokenizer _tokenizer;
    private int _disposed;

    public int SampleRate => _pipeline.SampleRate;

    /// <summary>The speaker encoder's expected reference sample rate (16 kHz).</summary>
    public int SpeakerSampleRate => _speaker.Config.SampleRate;

    public ZonosTts(ZonosConfig cfg, EspeakPhonemizer phonemizer, string language)
    {
        _cfg = cfg;
        _pipeline = new ZonosPipeline(cfg);
        _conditioning = new ZonosConditioning();
        _speaker = new ZonosSpeakerEncoder(ZonosSpeakerConfig.Default);
        _tokenizer = new ZonosPhonemeTokenizer(phonemizer, language);
    }

    /// <summary>Loads all weights: <paramref name="model"/> = the transformer safetensors (backbone + codebooks +
    /// prefix_conditioner), <paramref name="dac"/> = the DAC 44.1 kHz codec, <paramref name="speaker"/> /
    /// <paramref name="speakerLda"/> = the ResNet293 speaker model and its LDA head.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> model, IReadOnlyDictionary<string, Tensor> dac,
        IReadOnlyDictionary<string, Tensor> speaker, IReadOnlyDictionary<string, Tensor> speakerLda)
    {
        _pipeline.LoadWeights(model, dac);
        _conditioning.LoadWeights(model);
        _speaker.LoadWeights(speaker, speakerLda);
    }

    /// <summary>Synthesizes <paramref name="text"/> in the voice of <paramref name="refWav16k"/> (mono 16 kHz).</summary>
    public float[] Synthesize(IBackend backend, string text, ReadOnlySpan<float> refWav16k, ZonosControls controls,
        int seed = 0, Action<GenerationProgress>? progress = null)
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(ZonosTts));
        if (refWav16k.Length == 0) throw new ArgumentException("Zonos requires a voice-reference clip.", nameof(refWav16k));

        // Full-F32 GEMM end-to-end (speaker encoder → conditioner → AR loop). TF32's ~1e-3 error accumulates over
        // the autoregressive steps and degenerates generation (verified: TF32 → babble, F32 → reference parity).
        bool prevHighPrec = backend.HighPrecisionGemm;
        backend.HighPrecisionGemm = true;
        try
        {
            Tensor speaker = _speaker.EmbedFromWav(backend, refWav16k);
            int[] phonemes = _tokenizer.Tokenize(text);
            return SynthesizeCore(backend, phonemes, speaker, controls, seed, progress);
        }
        finally { backend.HighPrecisionGemm = prevHighPrec; }
    }

    /// <summary>Synthesizes from pre-computed phoneme ids (bypassing the tokenizer) — used by tests and callers
    /// with an external phonemization.</summary>
    public float[] SynthesizeFromPhonemes(IBackend backend, ReadOnlySpan<int> phonemes, ReadOnlySpan<float> refWav16k,
        ZonosControls controls, int seed = 0, Action<GenerationProgress>? progress = null)
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(ZonosTts));
        bool prevHighPrec = backend.HighPrecisionGemm;
        backend.HighPrecisionGemm = true;
        try
        {
            Tensor speaker = _speaker.EmbedFromWav(backend, refWav16k);
            return SynthesizeCore(backend, phonemes, speaker, controls, seed, progress);
        }
        finally { backend.HighPrecisionGemm = prevHighPrec; }
    }

    private float[] SynthesizeCore(IBackend backend, ReadOnlySpan<int> phonemes, Tensor speaker, ZonosControls controls,
        int seed, Action<GenerationProgress>? progress)
    {
        float[] emotion = NormalizeEmotion(controls.Emotion);
        Tensor cond = _conditioning.BuildPrefix(backend, phonemes, speaker, emotion, controls.Fmax,
            controls.PitchStd, controls.SpeakingRate, controls.LanguageId, conditional: true);
        Tensor uncond = _conditioning.BuildPrefix(backend, phonemes, speaker, emotion, controls.Fmax,
            controls.PitchStd, controls.SpeakingRate, controls.LanguageId, conditional: false);

        Logs.Info($"[Zonos] {phonemes.Length} phoneme tokens, cfg={controls.CfgScale}, temp={controls.Temperature}, seed={seed}.");
        float[] audio = _pipeline.Generate(backend, cond, uncond, controls.MaxNewTokens, seed,
            controls.CfgScale, controls.Temperature, controls.MinP, controls.RepetitionPenalty, progress);

        speaker.Dispose();
        cond.Dispose();
        uncond.Dispose();
        return audio;
    }

    /// <summary>The reference renormalizes the 8-way emotion vector to sum 1 (<c>make_cond_dict</c>).</summary>
    private static float[] NormalizeEmotion(float[] emotion)
    {
        float sum = 0;
        foreach (float e in emotion) sum += e;
        if (sum <= 0) return emotion;
        float[] outArr = new float[emotion.Length];
        for (int i = 0; i < emotion.Length; i++) outArr[i] = emotion[i] / sum;
        return outArr;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _pipeline.EnumerateWeights()) yield return t;
        foreach (Tensor t in _conditioning.EnumerateWeights()) yield return t;
        foreach (Tensor t in _speaker.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _pipeline.Dispose();
        _conditioning.Dispose();
        _speaker.Dispose();
        GC.SuppressFinalize(this);
    }
}
