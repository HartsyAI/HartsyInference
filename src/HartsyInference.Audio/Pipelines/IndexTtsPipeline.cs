using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.IndexTts;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Pipelines;

/// <summary>IndexTTS-1.5 zero-shot voice-cloning pipeline. Call graph (matches the reference <c>infer()</c>,
/// ground-truthed against the real checkpoint and upstream source — see
/// <c>docs/Research/INDEX_TTS_ARCHITECTURE.md</c> for the full trail):
/// <code>
///   refMel   = mel(referenceAudio)                                   # [1, T_ref, 100], reused for both paths below
///   prefix   = Perceiver(Conformer(refMel))                          # [1, 32, 1280], no position embedding
///   textIds  = tokenizer.Encode(CjkBoundaries(text))
///   latent   = T2sDecoder.Generate(prefix, textIds)                  # AR sample mel codes, keep final_norm'd hidden states
///   wav      = BigVgan(latent, refMel)                               # ECAPA d-vector + GPT-latent conditioning
/// </code>
/// No codec decode step: the DVAE checkpoint (<c>dvae.pth</c>) is confirmed unused at inference by the reference
/// <c>infer.py</c> (the whole load block is commented out upstream) — BigVGAN conditions on the GPT's own latent
/// hidden states directly, not a decoded mel.</summary>
public sealed class IndexTtsPipeline : IDisposable
{
    private readonly IndexTtsConfig _cfg;
    private readonly IndexTtsTokenizer _tokenizer;
    private readonly IndexTtsSpeakerEncoder _speakerEncoder;
    private readonly IndexTtsT2sDecoder _t2s;
    private readonly IndexTtsBigVganGenerator _bigVgan;
    private readonly MelSpectrogramExtractor _melExtractor;
    private readonly PytorchPickleLoader _gptLoader;
    private readonly PytorchPickleLoader _bigVganLoader;
    private int _disposed;

    public string ModelName => "indextts-1.5";

    private IndexTtsPipeline(IndexTtsConfig cfg, IndexTtsTokenizer tokenizer, IndexTtsSpeakerEncoder speakerEncoder,
        IndexTtsT2sDecoder t2s, IndexTtsBigVganGenerator bigVgan, PytorchPickleLoader gptLoader, PytorchPickleLoader bigVganLoader)
    {
        _cfg = cfg;
        _tokenizer = tokenizer;
        _speakerEncoder = speakerEncoder;
        _t2s = t2s;
        _bigVgan = bigVgan;
        _gptLoader = gptLoader;
        _bigVganLoader = bigVganLoader;
        // IndexTTS's own MelSpectrogramFeatures matches F5VocosConfig's torchaudio MelSpectrogram parameterization
        // exactly (24kHz, n_fft 1024, hop 256, 100 mels, HTK scale, magnitude spectrum, center padding, natural
        // log) EXCEPT the log floor: the reference calls safe_log(mel, clip_val=1e-7), not F5/Vocos's 1e-5 —
        // verified against the real index-tts source (utils/common.py's safe_log, utils/feature_extractors.py).
        _melExtractor = new MelSpectrogramExtractor(MelSpectrogramExtractor.F5VocosConfig() with { LogFloor = 1e-7f });
    }

    /// <summary>Loads IndexTTS-1.5 from already-downloaded checkpoint files. <paramref name="dvaePath"/> is
    /// accepted for completeness (the Engine layer downloads it alongside the rest of the HF repo) but is never
    /// read — confirmed unused in the real inference path.</summary>
    public static Task<IndexTtsPipeline> LoadAsync(string tokenizerPath, string gptPath, string bigVganPath,
        string? dvaePath, IndexTtsConfig? cfg, CancellationToken ct = default)
    {
        _ = dvaePath;
        ct.ThrowIfCancellationRequested();
        IndexTtsConfig resolved = cfg ?? IndexTtsConfig.V1_5;

        IndexTtsTokenizer? tokenizer = null;
        PytorchPickleLoader? gptLoader = null;
        IndexTtsSpeakerEncoder? speakerEncoder = null;
        IndexTtsT2sDecoder? t2s = null;
        PytorchPickleLoader? bigVganLoader = null;
        IndexTtsBigVganGenerator? bigVgan = null;
        try
        {
            tokenizer = new IndexTtsTokenizer(tokenizerPath);

            // The loaders must outlive this method: WhisperOps.EnsureF32 passes an already-F32 tensor through
            // unchanged (the caller does not own the result, per its own doc), so every weight the model classes
            // below retain is the SAME Tensor the loader owns. Disposing the loader here would free those tensors
            // out from under a pipeline that hasn't synthesized anything yet. Both loaders are kept as fields and
            // disposed only in IndexTtsPipeline.Dispose() on the success path — or right here, alongside every
            // other component already constructed, if a later step throws (a corrupt/incompatible checkpoint
            // must not leak the gigabytes already loaded).
            gptLoader = new PytorchPickleLoader();
            gptLoader.Load(gptPath, recursiveFlatten: true);
            IReadOnlyDictionary<string, Tensor> gptWeights = gptLoader.GetAllTensors();

            speakerEncoder = new IndexTtsSpeakerEncoder(resolved.ConditioningEncoder, resolved.Gpt.Hidden);
            speakerEncoder.LoadWeights(gptWeights, "model.conditioning_encoder", "model.perceiver_encoder");

            t2s = new IndexTtsT2sDecoder(resolved.Gpt, resolved.MaxMelTokens);
            t2s.LoadWeights(gptWeights);

            bigVganLoader = new PytorchPickleLoader();
            bigVganLoader.Load(bigVganPath, recursiveFlatten: true);
            IReadOnlyDictionary<string, Tensor> bigVganWeights = bigVganLoader.GetAllTensors();
            bigVgan = new IndexTtsBigVganGenerator(resolved.BigVgan);
            bigVgan.LoadWeights(bigVganWeights, "generator", "generator.speaker_encoder");

            return Task.FromResult(new IndexTtsPipeline(resolved, tokenizer, speakerEncoder, t2s, bigVgan, gptLoader, bigVganLoader));
        }
        catch
        {
            bigVgan?.Dispose();
            t2s?.Dispose();
            speakerEncoder?.Dispose();
            bigVganLoader?.Dispose();
            gptLoader?.Dispose();
            tokenizer?.Dispose();
            throw;
        }
    }

    /// <summary>Clones the voice in <paramref name="referenceAudioMono"/> speaking <paramref name="text"/>.
    /// Returns 24 kHz mono PCM float samples in [-1, 1].</summary>
    public float[] Synthesize(IBackend backend, string text, float[] referenceAudioMono, int referenceSampleRate, IndexTtsOptions? options = null)
    {
        ThrowIfDisposed();
        IndexTtsOptions opts = options ?? new IndexTtsOptions();

        // Validate text (cheap, no allocation) before computing the reference mel, so a rejected prompt never
        // leaves an undisposed Tensor behind.
        string normalized = IndexTtsTextNormalizer.InjectCjkBoundaries(text);
        int[] textIds = _tokenizer.Encode(normalized);
        if (textIds.Length == 0) throw new ArgumentException("Text produced zero tokens after tokenization.", nameof(text));
        if (textIds.Length > _cfg.MaxTextTokens)
            throw new ArgumentException($"Text has {textIds.Length} tokens, exceeding IndexTTS-1.5's max_text_tokens ({_cfg.MaxTextTokens}).", nameof(text));

        float[] refAt24k = referenceSampleRate == _cfg.SampleRate
            ? referenceAudioMono
            : Resampler.Create(referenceSampleRate, _cfg.SampleRate).Resample(referenceAudioMono);
        (Tensor refMel, int refMelLen) = ComputeMelChannelsLast(refAt24k);

        Tensor prefix = _speakerEncoder.Forward(backend, refMel, refMelLen);

        Random rng = new(unchecked((int)opts.Seed));
        Tensor latent = _t2s.Generate(backend, prefix, textIds, opts, rng);
        prefix.Dispose();

        Tensor wave = _bigVgan.Forward(backend, latent, (int)latent.Shape[1], refMel, refMelLen);
        latent.Dispose();
        refMel.Dispose();

        int n = (int)wave.ElementCount;
        float[] pcm = new float[n];
        unsafe
        {
            float* wp = (float*)wave.DataPointer;
            for (int i = 0; i < n; i++) pcm[i] = wp[i];
        }
        wave.Dispose();
        return pcm;
    }

    /// <summary>Runs the shared mel front end and transposes its channels-first <c>[nMels, T]</c> output into the
    /// channels-last <c>[1, T, nMels]</c> layout <see cref="IndexTtsConformerEncoder"/>/<see cref="IndexTtsEcapaTdnn"/> expect.</summary>
    private (Tensor, int) ComputeMelChannelsLast(float[] audio)
    {
        float[,] mel = _melExtractor.Compute(audio);
        int nMels = mel.GetLength(0), t = mel.GetLength(1);
        Tensor outT = new(new TensorShape(1, t, nMels), DType.F32);
        unsafe
        {
            float* op = (float*)outT.DataPointer;
            for (int f = 0; f < t; f++)
                for (int m = 0; m < nMels; m++)
                    op[(long)f * nMels + m] = mel[m, f];
        }
        return (outT, t);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _speakerEncoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _t2s.EnumerateWeights()) yield return t;
        foreach (Tensor t in _bigVgan.EnumerateWeights()) yield return t;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(IndexTtsPipeline));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _tokenizer.Dispose();
        _speakerEncoder.Dispose();
        _t2s.Dispose();
        _bigVgan.Dispose();
        // The loaders own the actual weight-tensor memory every component above holds references into (see
        // LoadAsync's remarks); dispose them last so nothing above touches freed tensors mid-teardown.
        _gptLoader.Dispose();
        _bigVganLoader.Dispose();
        GC.SuppressFinalize(this);
    }
}
