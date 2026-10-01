using System.Text;
using System.Text.Json;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Diagnostics;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Preprocessing;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Pipelines;

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
/// <para>Greedy decoding only in v1. Beam search, temperature fallback, and long-form
/// audio (sequential or chunked) land in subsequent passes; the encoder and decoder
/// modules already expose the necessary state hooks. <see cref="SegmentAudio"/> gives
/// SEGMENT-level timestamps from the model's native <c>&lt;|t|&gt;</c> tokens; true
/// word-level alignment is not available because it needs cross-attention DTW and
/// <c>WhisperDecoder</c> runs fused SDPA without surfacing attention weights.</para></summary>
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

    /// <summary>Whether the loaded checkpoint uses the multilingual token layout. English-only (<c>*.en</c>) checkpoints prompt with SOT alone, so <see cref="WhisperOptions.Language"/> and <see cref="WhisperOptions.Translate"/> have no effect on them.</summary>
    public bool IsMultilingual => _tokenizer.IsMultilingual;

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

    /// <summary>Files a Whisper repo contributes. Load and prefetch share this list, so installing a variant
    /// fetches exactly what loading it will ask for. The weights sit last because their presence is what marks
    /// the model installed — see <see cref="AudioModelCache.FetchAllAsync"/>.</summary>
    public static IReadOnlyList<AudioModelFile> ModelFiles { get; } =
    [
        new("vocab.json"),
        new("merges.txt"),
        new("config.json"),
        // Every OpenAI and distil release ships these; optional so a hand-built directory still loads.
        new("added_tokens.json", Required: false),
        new("tokenizer_config.json", Required: false),
        new("generation_config.json", Required: false),
        new("model.safetensors"),
    ];

    /// <summary>Loads a Whisper pipeline from a HuggingFace repo, downloading <see cref="ModelFiles"/> into the
    /// shared cache on first use.</summary>
    /// <param name="hfRepoId">Repo id, e.g. <c>"openai/whisper-tiny"</c>.</param>
    /// <param name="cfg">Override config, used as given; if null we infer it from the repo name
    /// (works for the standard OpenAI / distil-whisper releases) and take the vocabulary size from the
    /// checkpoint's own <c>config.json</c>.</param>
    public static async Task<WhisperPipeline> LoadAsync(
        string hfRepoId,
        WhisperConfig? cfg = null,
        CancellationToken ct = default)
    {
        // Infer before fetching so an unknown repo fails without downloading anything.
        WhisperConfig resolvedCfg = cfg ?? InferConfig(hfRepoId);

        string repoDir = AudioModelCache.GetRepoDirectory(hfRepoId, "stt");
        IReadOnlyDictionary<string, string> fetched = await AudioModelCache
            .FetchAllAsync(hfRepoId, ModelFiles, category: "stt", ct: ct).ConfigureAwait(false);
        if (cfg is null) resolvedCfg = ApplyCheckpointVocab(resolvedCfg, fetched["config.json"]);

        SafeTensorsLoader loader = new();
        loader.Load(fetched["model.safetensors"]);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();

        WhisperEncoder encoder = new(resolvedCfg);
        WhisperDecoder decoder = new(resolvedCfg);
        encoder.LoadWeights(weights);
        decoder.LoadWeights(weights);

        WhisperTokenizer tokenizer = new(repoDir);
        RequireSameLayout(resolvedCfg, tokenizer, repoDir);
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
        return _tokenizer.Decode(TranscribeTokenIds(backend, audio, sampleRate, options ?? new WhisperOptions()).ToArray());
    }

    /// <summary>The ids <see cref="TranscribeAudio"/> decodes (timestamp tokens included when
    /// <see cref="WhisperOptions.WithTimestamps"/> is set), for token-level comparisons across builds.</summary>
    internal List<int> TranscribeTokenIds(IBackend backend, float[] audio, int sampleRate, WhisperOptions options)
    {
        ThrowIfDisposed();
        StageTimer? timer = StageTimer.Start(backend, "Whisper");
        (Tensor mel, double _) = PrepareMel(audio, sampleRate);
        timer?.Mark("mel");
        return DecodeTokens(backend, mel, options, timer);
    }

    /// <summary>The text <see cref="TranscribeAudio"/> returns for <paramref name="ids"/>.</summary>
    internal string DecodeText(List<int> ids) => _tokenizer.Decode(ids.ToArray());

    /// <summary>Transcribes with Whisper's native <c>&lt;|t|&gt;</c> timestamp tokens enabled and returns the decoded
    /// spans. <b>Segment granularity, not word granularity</b> — see <see cref="WhisperSegment"/>. Like
    /// <see cref="TranscribeAudio"/> this covers only the first 30 s of the clip (single-chunk decode); longer audio
    /// is silently clipped, so no segment can start past 30 s. Returns an empty list when greedy decoding produced no
    /// well-formed timestamp pair (the decoder does not enforce OpenAI's timestamp rules).</summary>
    public IReadOnlyList<WhisperSegment> SegmentAudio(IBackend backend, float[] audio, int sampleRate, WhisperOptions? options = null)
    {
        ThrowIfDisposed();
        WhisperOptions opts = options ?? new WhisperOptions();
        StageTimer? timer = StageTimer.Start(backend, "Whisper");
        (Tensor mel, double seconds) = PrepareMel(audio, sampleRate);
        timer?.Mark("mel");
        return ParseSegments(DecodeTokens(backend, mel, opts with { WithTimestamps = true }, timer), seconds);
    }

    /// <summary>Timestamped decode from a pre-computed mel; <paramref name="audioSeconds"/> only closes a span the
    /// model left open. Mirrors <see cref="TranscribeFromMel"/> for tests that inject their own mel.</summary>
    public IReadOnlyList<WhisperSegment> SegmentFromMel(IBackend backend, float[,] mel, double audioSeconds, WhisperOptions options)
    {
        ThrowIfDisposed();
        List<int> tokens = DecodeTokens(backend, MelToTensor(mel), options with { WithTimestamps = true }, StageTimer.Start(backend, "Whisper"));
        return ParseSegments(tokens, audioSeconds);
    }

    /// <summary>Resamples to 16 kHz, zero-pads/clips to the 30 s chunk, and returns the <c>[1, n_mels, n_frames]</c>
    /// mel (owned by the caller) plus the pre-pad duration. The padding is never materialized: frames that lie wholly
    /// in it are a constant the extractor fills without transforming.</summary>
    private (Tensor Mel, double Seconds) PrepareMel(float[] audio, int sampleRate)
    {
        float[] mono16k = audio;
        if (sampleRate != 16_000)
        {
            Resampler resampler = Resampler.Create(sampleRate, 16_000);
            mono16k = resampler.Resample(audio);
        }

        // Zero-pad / clip to exactly 30 s (480 000 samples).
        const int n30s = 30 * 16_000;
        int copyLen = Math.Min(mono16k.Length, n30s);
        int frames = _melExtractor.OutputFrames(n30s);
        Tensor mel = new(new TensorShape(1, _cfg.NumMelBins, frames), DType.F32);
        try
        {
            _melExtractor.ComputeZeroPadded(mono16k.AsSpan(0, copyLen), n30s, mel.AsSpan<float>());
        }
        catch
        {
            mel.Dispose();
            throw;
        }
        return (mel, copyLen / 16_000.0);
    }

    /// <summary>Lower-level entry point that takes a pre-computed mel spectrogram of
    /// shape <c>[n_mels, n_frames]</c>. Exposed for tests that want to bypass the audio
    /// preprocessing path and inject a hand-crafted mel.</summary>
    public string TranscribeFromMel(IBackend backend, float[,] mel, WhisperOptions options)
    {
        ThrowIfDisposed();
        return _tokenizer.Decode(DecodeTokens(backend, MelToTensor(mel), options, StageTimer.Start(backend, "Whisper")).ToArray());
    }

    /// <summary>Encoder forward + greedy decode over a <c>[1, n_mels, n_frames]</c> mel it takes ownership of, returning
    /// the raw generated ids (timestamp tokens included when <see cref="WhisperOptions.WithTimestamps"/> is set).</summary>
    private List<int> DecodeTokens(IBackend backend, Tensor mel, WhisperOptions options, StageTimer? timer)
    {
        int nFrames = (int)mel.Shape[2];
        Tensor encoded;
        try
        {
            encoded = _encoder.Forward(backend, mel, timer);
        }
        finally { mel.Dispose(); }
        timer?.Mark("encoder");

        try
        {
            using WhisperDecoder.DecodeState state = _decoder.StartDecode(backend, encoded);
            timer?.Mark("crossKv");
            state.Timer = timer;
            List<int> tokens = GreedyDecode(backend, state, options, timer);
            timer?.Report($"frames={nFrames} tokens={tokens.Count}");
            return tokens;
        }
        finally { encoded.Dispose(); }
    }

    /// <summary>Splits the generated ids on <c>&lt;|t|&gt;</c> tokens into timestamped spans. Times are reported
    /// verbatim from the tokens (never clamped or interpolated); only a span the model left open is closed, at
    /// <paramref name="audioSeconds"/>.</summary>
    private List<WhisperSegment> ParseSegments(List<int> tokens, double audioSeconds)
    {
        List<WhisperSegment> segments = [];
        List<int> buffer = [];
        double? start = null;
        foreach (int id in tokens)
        {
            if (!_tokenizer.IsTimestampId(id))
            {
                buffer.Add(id);
                continue;
            }
            double at = _tokenizer.SecondsForTimestamp(id);
            if (start is not null && buffer.Count > 0)
            {
                Emit(segments, buffer, start.Value, at);
            }
            buffer.Clear();
            start = at;
        }
        if (start is not null && buffer.Count > 0)
        {
            Emit(segments, buffer, start.Value, Math.Max(start.Value, Math.Min(audioSeconds, 30.0)));
        }
        return segments;
    }

    /// <summary>Decodes one buffered span and appends it when it has non-empty text and a non-negative duration.</summary>
    private void Emit(List<WhisperSegment> segments, List<int> buffer, double start, double end)
    {
        string text = _tokenizer.Decode(buffer.ToArray()).Trim();
        if (text.Length > 0 && end >= start)
        {
            segments.Add(new WhisperSegment { Text = text, Start = start, End = end });
        }
    }

    private List<int> GreedyDecode(IBackend backend, WhisperDecoder.DecodeState state, WhisperOptions opts, StageTimer? timer)
    {
        // Prompt: [SOT, <|lang|>, transcribe|translate, <|notimestamps|>?] — or [SOT, <|notimestamps|>?] English-only.
        int[] prompt = _tokenizer.BuildPromptIds(opts.Language, opts.Translate, opts.WithTimestamps);

        // Run the prompt through the decoder in a single pass — the logits at the last
        // prompt position are the distribution we sample our first text token from.
        Tensor logits = _decoder.DecodeStep(backend, prompt, state);
        int nextToken = ArgMaxIgnoringSpecial(logits, _tokenizer, timer);
        logits.Dispose();
        timer?.Accumulate("step.argmax");
        timer?.Mark("prompt");

        List<int> generated = new(opts.MaxNewTokens);
        int[] singleBuf = new int[1];
        for (int step = 0; step < opts.MaxNewTokens; step++)
        {
            if (nextToken == _tokenizer.EotId) break;
            generated.Add(nextToken);

            // Feed only the new token forward; the cache holds everything prior.
            singleBuf[0] = nextToken;
            Tensor stepLogits = _decoder.DecodeStep(backend, singleBuf, state);
            nextToken = ArgMaxIgnoringSpecial(stepLogits, _tokenizer, timer);
            stepLogits.Dispose();
            timer?.Accumulate("step.argmax");
        }
        timer?.Mark($"decode[{generated.Count} steps]");

        return generated;
    }

    /// <summary>Greedy argmax with the standard Whisper suppress list applied. We
    /// mask the prompt special tokens (SOT, language tags, task tokens, no-timestamps,
    /// no-speech) so they can't be emitted mid-stream; otherwise the model often
    /// re-emits its own start tokens and freezes the decode. The full OpenAI suppress
    /// list (~99 entries including punctuation heuristics) lands with temperature
    /// fallback later. Ids come from the tokenizer, not the config — only it knows
    /// whether this checkpoint uses the v3 (100-language) or the English-only layout.</summary>
    private static unsafe int ArgMaxIgnoringSpecial(Tensor logits, WhisperTokenizer tokenizer, StageTimer? timer)
    {
        int vocab = (int)logits.Shape[logits.Shape.Rank - 1];
        float* p = (float*)logits.DataPointer;

        // Suppress prompt-only tokens: SOT, every language tag, and both task tokens.
        for (int id = tokenizer.SotId; id <= tokenizer.TranscribeId; id++)
            if (id < vocab) p[id] = float.NegativeInfinity;
        if (tokenizer.NoSpeechId < vocab) p[tokenizer.NoSpeechId] = float.NegativeInfinity;
        if (tokenizer.NoTimestampsId < vocab) p[tokenizer.NoTimestampsId] = float.NegativeInfinity;

        int best = 0;
        float bestV = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            float val = p[v];
            if (val > bestV) { bestV = val; best = v; }
        }
        if (timer is not null)
        {
            float second = float.NegativeInfinity;
            for (int v = 0; v < vocab; v++)
            {
                if (v != best && p[v] > second) second = p[v];
            }
            timer.NoteMinimum("top1-top2 logit margin", bestV - second);
        }
        return best;
    }

    /// <summary>Wraps a caller's 2-D <c>[n_mels, n_frames]</c> mel into the <c>[1, n_mels, n_frames]</c> tensor the encoder takes.</summary>
    private unsafe Tensor MelToTensor(float[,] mel)
    {
        int nMels = mel.GetLength(0);
        int nFrames = mel.GetLength(1);
        if (nMels != _cfg.NumMelBins)
            throw new ArgumentException($"mel has {nMels} bins but model expects {_cfg.NumMelBins}");
        TensorShape shape = new(1, nMels, nFrames);
        Tensor t = new(shape, DType.F32);
        float* dst = (float*)t.DataPointer;
        for (int m = 0; m < nMels; m++)
            for (int f = 0; f < nFrames; f++)
                dst[m * nFrames + f] = mel[m, f];
        return t;
    }

    /// <summary>Infers a <see cref="WhisperConfig"/> from a HuggingFace repo name. Covers
    /// the official OpenAI releases (multilingual and <c>.en</c>) plus the distil-whisper variants.
    /// Falls back to a load from the repo's <c>config.json</c> for unrecognized names (future work).</summary>
    public static WhisperConfig InferConfig(string hfRepoId)
    {
        string lower = hfRepoId.ToLowerInvariant();
        return lower switch
        {
            "openai/whisper-tiny" => WhisperConfig.Tiny,
            "openai/whisper-tiny.en" => WhisperConfig.TinyEn,
            "openai/whisper-base" => WhisperConfig.Base,
            "openai/whisper-base.en" => WhisperConfig.BaseEn,
            "openai/whisper-small" => WhisperConfig.Small,
            "openai/whisper-small.en" => WhisperConfig.SmallEn,
            "openai/whisper-medium" => WhisperConfig.Medium,
            "openai/whisper-medium.en" => WhisperConfig.MediumEn,
            "openai/whisper-large-v2" => WhisperConfig.LargeV2,
            "openai/whisper-large-v3" => WhisperConfig.LargeV3,
            "openai/whisper-large-v3-turbo" => WhisperConfig.LargeV3Turbo,
            "distil-whisper/distil-large-v2" => WhisperConfig.DistilLargeV2,
            "distil-whisper/distil-large-v3" => WhisperConfig.DistilLargeV3,
            "distil-whisper/distil-large-v3.5" => WhisperConfig.DistilLargeV3_5,
            "distil-whisper/distil-medium.en" => WhisperConfig.DistilMediumEn,
            "distil-whisper/distil-small.en" => WhisperConfig.DistilSmallEn,
            _ => throw new ArgumentException(
                $"Unknown Whisper repo '{hfRepoId}'. Pass an explicit WhisperConfig to LoadAsync.",
                nameof(hfRepoId)),
        };
    }

    /// <summary>Replaces the preset's vocabulary size with the checkpoint's own <c>vocab_size</c> when the two differ: the embedding row count is a checkpoint fact (51864 English-only, 51865 multilingual, 51866 v3+), and the decoder refuses to load a mismatch. Only the vocabulary facts (<see cref="WhisperConfig.VocabSize"/> and <see cref="WhisperConfig.IsMultilingual"/>) come from disk; the layer shape and <see cref="WhisperConfig.LanguageCount"/> stay with the preset.</summary>
    private static WhisperConfig ApplyCheckpointVocab(WhisperConfig cfg, string configJsonPath)
    {
        using FileStream fs = File.OpenRead(configJsonPath);
        using JsonDocument doc = JsonDocument.Parse(fs);
        if (!doc.RootElement.TryGetProperty("vocab_size", out JsonElement vocab)
            || !vocab.TryGetInt32(out int vocabSize) || vocabSize == cfg.VocabSize)
        {
            return cfg;
        }
        return cfg with { VocabSize = vocabSize, IsMultilingual = vocabSize >= WhisperTokenizer.MultilingualVocabSize };
    }

    /// <summary>The config and the tokenizer decide the layout from different files (<c>config.json</c> versus <c>vocab.json</c> + <c>added_tokens.json</c>); the prompt follows the tokenizer, so a directory whose files disagree is refused rather than decoded with a prompt the weights were not trained on.</summary>
    private static void RequireSameLayout(WhisperConfig cfg, WhisperTokenizer tokenizer, string repoDir)
    {
        if (cfg.IsMultilingual == tokenizer.IsMultilingual) return;
        throw new InvalidOperationException(
            $"'{repoDir}': config.json says {(cfg.IsMultilingual ? "multilingual" : "English-only")} (vocab {cfg.VocabSize}) " +
            $"but the tokenizer files resolve to {(tokenizer.IsMultilingual ? "multilingual" : "English-only")} (vocab {tokenizer.VocabSize}).");
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
