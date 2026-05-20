using System.Text;
using SharpInference.Audio.Cache;
using SharpInference.Audio.Io;
using SharpInference.Audio.Models.Whisper;
using SharpInference.Audio.Preprocessing;
using SharpInference.Core.Backends;
using SharpInference.Core.Pipelines;
using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tokenizers;

namespace SharpInference.Audio.Pipelines;

/// <summary>End-to-end Whisper STT pipeline: raw audio → mel → encoder → autoregressive
/// decoder → text. Owns the encoder, decoder, and tokenizer for the lifetime of one
/// model load. Backends are passed in per call so the same loaded pipeline can run on
/// CPU during testing and CUDA in production.
///
/// <para><b>Usage:</b>
/// <code>
/// using WhisperPipeline pipe = await WhisperPipeline.LoadAsync(backend, "openai/whisper-tiny");
/// string text = pipe.TranscribeWav("input.wav");
/// </code></para>
///
/// <para>Greedy decoding only in v1. Beam search, temperature fallback, long-form
/// audio (sequential or chunked), and word-level timestamps land in subsequent
/// passes; the encoder and decoder modules already expose the necessary state hooks.</para></summary>
public sealed class WhisperPipeline : IAudioPipeline, IDisposable
{
    private readonly WhisperConfig _cfg;
    private readonly WhisperEncoder _encoder;
    private readonly WhisperDecoder _decoder;
    private readonly WhisperTokenizer _tokenizer;
    private readonly MelSpectrogramExtractor _melExtractor;
    private readonly SafeTensorsLoader _loader;
    private int _disposed;

    /// <summary>Name of the loaded model (HF repo id).</summary>
    public string ModelName { get; }

    private WhisperPipeline(
        string modelName,
        WhisperConfig cfg,
        WhisperEncoder encoder,
        WhisperDecoder decoder,
        WhisperTokenizer tokenizer,
        SafeTensorsLoader loader)
    {
        ModelName = modelName;
        _cfg = cfg;
        _encoder = encoder;
        _decoder = decoder;
        _tokenizer = tokenizer;
        _loader = loader;
        _melExtractor = new MelSpectrogramExtractor(MelSpectrogramExtractor.WhisperConfig(cfg.NumMelBins));
    }

    /// <summary>Loads a Whisper pipeline from a HuggingFace repo. Downloads
    /// <c>model.safetensors</c>, <c>vocab.json</c>, <c>merges.txt</c>, and (where
    /// applicable) <c>added_tokens.json</c> into the shared cache on first use.</summary>
    /// <param name="hfRepoId">Repo id, e.g. <c>"openai/whisper-tiny"</c>.</param>
    /// <param name="cfg">Override config; if null we infer it from the repo name
    /// (works for the standard OpenAI / distil-whisper releases).</param>
    public static async Task<WhisperPipeline> LoadAsync(
        string hfRepoId,
        WhisperConfig? cfg = null,
        CancellationToken ct = default)
    {
        WhisperConfig resolvedCfg = cfg ?? InferConfig(hfRepoId);

        string repoDir = AudioModelCache.GetRepoDirectory(hfRepoId);
        Task<string> safetensors = AudioModelCache.GetAsync(hfRepoId, "model.safetensors", ct: ct);
        Task<string> vocab = AudioModelCache.GetAsync(hfRepoId, "vocab.json", ct: ct);
        Task<string> merges = AudioModelCache.GetAsync(hfRepoId, "merges.txt", ct: ct);
        Task<string> config = AudioModelCache.GetAsync(hfRepoId, "config.json", ct: ct);
        // added_tokens is optional — only multilingual checkpoints ship it.
        Task<string> addedTokens = TryGetOptional(hfRepoId, "added_tokens.json", ct);
        Task<string> tokenizerConfig = TryGetOptional(hfRepoId, "tokenizer_config.json", ct);
        Task<string> generationConfig = TryGetOptional(hfRepoId, "generation_config.json", ct);
        await Task.WhenAll(safetensors, vocab, merges, config, addedTokens, tokenizerConfig, generationConfig).ConfigureAwait(false);

        SafeTensorsLoader loader = new();
        loader.Load(safetensors.Result);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();

        WhisperEncoder encoder = new(resolvedCfg);
        WhisperDecoder decoder = new(resolvedCfg);
        encoder.LoadWeights(weights);
        decoder.LoadWeights(weights);

        WhisperTokenizer tokenizer = new(repoDir);
        return new WhisperPipeline(hfRepoId, resolvedCfg, encoder, decoder, tokenizer, loader);
    }

    /// <summary>Convenience: transcribe a WAV file. Auto-resamples to 16 kHz and
    /// downmixes to mono.</summary>
    public string TranscribeWav(IBackend backend, string wavPath, WhisperOptions? options = null)
    {
        WavFile.DecodedAudio audio = WavFile.Read(wavPath);
        return TranscribeAudio(backend, audio.ToMono(), audio.SampleRate, options);
    }

    /// <summary>Transcribes an in-memory mono audio buffer at the given sample rate.
    /// Resamples to 16 kHz internally if needed; zero-pads to the 30-second chunk
    /// length Whisper expects.</summary>
    public string TranscribeAudio(IBackend backend, float[] audio, int sampleRate, WhisperOptions? options = null)
    {
        ThrowIfDisposed();
        WhisperOptions opts = options ?? new WhisperOptions();

        // 1. Resample to 16 kHz if needed.
        float[] mono16k = audio;
        if (sampleRate != 16_000)
        {
            Resampler resampler = Resampler.Create(sampleRate, 16_000);
            mono16k = resampler.Resample(audio);
        }

        // 2. Zero-pad / clip to exactly 30 s (480 000 samples).
        const int n30s = 30 * 16_000;
        float[] padded = new float[n30s];
        int copyLen = Math.Min(mono16k.Length, n30s);
        Array.Copy(mono16k, padded, copyLen);

        // 3. Mel spectrogram.
        float[,] mel = _melExtractor.Compute(padded);

        // 4. Run pipeline.
        return TranscribeFromMel(backend, mel, opts);
    }

    /// <summary>Lower-level entry point that takes a pre-computed mel spectrogram of
    /// shape <c>[n_mels, n_frames]</c>. Exposed for tests that want to bypass the audio
    /// preprocessing path and inject a hand-crafted mel.</summary>
    public string TranscribeFromMel(IBackend backend, float[,] mel, WhisperOptions options)
    {
        ThrowIfDisposed();
        int nMels = mel.GetLength(0);
        int nFrames = mel.GetLength(1);
        if (nMels != _cfg.NumMelBins)
            throw new ArgumentException($"mel has {nMels} bins but model expects {_cfg.NumMelBins}");

        // Wrap the 2-D mel into a [1, n_mels, n_frames] Tensor.
        Tensor melTensor = MelToTensor(mel, nMels, nFrames);
        Tensor encoded;
        try
        {
            encoded = _encoder.Forward(backend, melTensor);
        }
        finally { melTensor.Dispose(); }

        try
        {
            using WhisperDecoder.DecodeState state = _decoder.StartDecode(backend, encoded);
            return GreedyDecode(backend, state, options);
        }
        finally { encoded.Dispose(); }
    }

    private string GreedyDecode(IBackend backend, WhisperDecoder.DecodeState state, WhisperOptions opts)
    {
        // Prompt: [SOT, <|lang|>, transcribe|translate, <|notimestamps|>?].
        int[] prompt = _tokenizer.BuildPromptIds(opts.Language, opts.Translate, opts.WithTimestamps);

        // Run the prompt through the decoder in a single pass — the logits at the last
        // prompt position are the distribution we sample our first text token from.
        Tensor logits = _decoder.DecodeStep(backend, prompt, state);
        int nextToken = ArgMaxIgnoringSpecial(logits, _cfg);
        logits.Dispose();

        List<int> generated = new(opts.MaxNewTokens);
        int[] singleBuf = new int[1];
        for (int step = 0; step < opts.MaxNewTokens; step++)
        {
            if (nextToken == _cfg.EndOfTextTokenId) break;
            generated.Add(nextToken);

            // Feed only the new token forward; the cache holds everything prior.
            singleBuf[0] = nextToken;
            Tensor stepLogits = _decoder.DecodeStep(backend, singleBuf, state);
            nextToken = ArgMaxIgnoringSpecial(stepLogits, _cfg);
            stepLogits.Dispose();
        }

        return _tokenizer.Decode(generated.ToArray());
    }

    /// <summary>Greedy argmax with the standard Whisper suppress list applied. We
    /// mask the prompt special tokens (SOT, language tags, task tokens, no-timestamps,
    /// no-speech) so they can't be emitted mid-stream; otherwise the model often
    /// re-emits its own start tokens and freezes the decode. The full OpenAI suppress
    /// list (~99 entries including punctuation heuristics) lands with temperature
    /// fallback later.</summary>
    private static unsafe int ArgMaxIgnoringSpecial(Tensor logits, WhisperConfig cfg)
    {
        int vocab = (int)logits.Shape[logits.Shape.Rank - 1];
        float* p = (float*)logits.DataPointer;

        // Suppress prompt-only tokens.
        for (int id = cfg.StartOfTranscriptTokenId; id <= cfg.TranscribeTokenId; id++)
            if (id < vocab) p[id] = float.NegativeInfinity;
        if (cfg.NoSpeechTokenId < vocab) p[cfg.NoSpeechTokenId] = float.NegativeInfinity;
        if (cfg.NoTimestampsTokenId < vocab) p[cfg.NoTimestampsTokenId] = float.NegativeInfinity;

        int best = 0;
        float bestV = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            float val = p[v];
            if (val > bestV) { bestV = val; best = v; }
        }
        return best;
    }

    private static unsafe Tensor MelToTensor(float[,] mel, int nMels, int nFrames)
    {
        TensorShape shape = new(1, nMels, nFrames);
        Tensor t = new(shape, DType.F32);
        float* dst = (float*)t.DataPointer;
        for (int m = 0; m < nMels; m++)
            for (int f = 0; f < nFrames; f++)
                dst[m * nFrames + f] = mel[m, f];
        return t;
    }

    private static async Task<string> TryGetOptional(string hfRepoId, string filename, CancellationToken ct)
    {
        try { return await AudioModelCache.GetAsync(hfRepoId, filename, ct: ct).ConfigureAwait(false); }
        catch (FileNotFoundException) { return string.Empty; }
    }

    /// <summary>Infers a <see cref="WhisperConfig"/> from a HuggingFace repo name. Covers
    /// the official OpenAI releases plus the distil-whisper variants. Falls back to a
    /// load from the repo's <c>config.json</c> for unrecognized names (future work).</summary>
    public static WhisperConfig InferConfig(string hfRepoId)
    {
        string lower = hfRepoId.ToLowerInvariant();
        return lower switch
        {
            "openai/whisper-tiny" or "openai/whisper-tiny.en" => WhisperConfig.Tiny,
            "openai/whisper-base" or "openai/whisper-base.en" => WhisperConfig.Base,
            "openai/whisper-small" or "openai/whisper-small.en" => WhisperConfig.Small,
            "openai/whisper-medium" or "openai/whisper-medium.en" => WhisperConfig.Medium,
            "openai/whisper-large-v2" => WhisperConfig.LargeV2,
            "openai/whisper-large-v3" => WhisperConfig.LargeV3,
            "openai/whisper-large-v3-turbo" => WhisperConfig.LargeV3Turbo,
            "distil-whisper/distil-large-v2" => WhisperConfig.DistilLargeV2,
            "distil-whisper/distil-large-v3" => WhisperConfig.DistilLargeV3,
            "distil-whisper/distil-medium.en" => WhisperConfig.DistilMediumEn,
            "distil-whisper/distil-small.en" => WhisperConfig.DistilSmallEn,
            _ => throw new ArgumentException(
                $"Unknown Whisper repo '{hfRepoId}'. Pass an explicit WhisperConfig to LoadAsync.",
                nameof(hfRepoId)),
        };
    }

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(WhisperPipeline));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _encoder.Dispose();
            _decoder.Dispose();
            _tokenizer.Dispose();
            _loader.Dispose();
        }
    }
}
