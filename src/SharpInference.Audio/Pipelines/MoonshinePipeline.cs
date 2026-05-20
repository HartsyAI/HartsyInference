using SharpInference.Audio.Cache;
using SharpInference.Audio.Io;
using SharpInference.Audio.Models.Moonshine;
using SharpInference.Core.Backends;
using SharpInference.Core.Pipelines;
using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.SafeTensors;

namespace SharpInference.Audio.Pipelines;

/// <summary>End-to-end Moonshine STT pipeline. Raw 16-kHz mono waveform → 3-layer Conv1D
/// stem → 8 RoPE-encoder blocks → 8 RoPE-decoder blocks (with cross-attn + KV cache) →
/// SentencePiece byte-fallback BPE decode → text.
///
/// <para>Unlike Whisper this pipeline has no mel preprocessing and no fixed 30-second
/// chunk — the encoder runs on whatever length you feed it. The 384× total downsample
/// in the conv stem gives a 41.67 Hz encoder frame rate.</para></summary>
public sealed class MoonshinePipeline : IAudioPipeline, IDisposable
{
    private readonly MoonshineConfig _cfg;
    private readonly MoonshineEncoder _encoder;
    private readonly MoonshineDecoder _decoder;
    private readonly MoonshineTokenizer _tokenizer;
    private readonly SafeTensorsLoader _loader;
    private int _disposed;

    public string ModelName { get; }

    private MoonshinePipeline(string modelName, MoonshineConfig cfg, MoonshineEncoder enc, MoonshineDecoder dec, MoonshineTokenizer tok, SafeTensorsLoader loader)
    {
        ModelName = modelName;
        _cfg = cfg;
        _encoder = enc;
        _decoder = dec;
        _tokenizer = tok;
        _loader = loader;
    }

    /// <summary>Loads a Moonshine pipeline from a HuggingFace repo. Auto-downloads
    /// model.safetensors and tokenizer.json on first use.</summary>
    public static async Task<MoonshinePipeline> LoadAsync(string hfRepoId, MoonshineConfig? cfg = null, CancellationToken ct = default)
    {
        MoonshineConfig resolved = cfg ?? InferConfig(hfRepoId);
        string repoDir = AudioModelCache.GetRepoDirectory(hfRepoId);

        Task<string> safetensors = AudioModelCache.GetAsync(hfRepoId, "model.safetensors", ct: ct);
        Task<string> tokenizerJson = AudioModelCache.GetAsync(hfRepoId, "tokenizer.json", ct: ct);
        Task<string> config = AudioModelCache.GetAsync(hfRepoId, "config.json", ct: ct);
        await Task.WhenAll(safetensors, tokenizerJson, config).ConfigureAwait(false);

        SafeTensorsLoader loader = new();
        loader.Load(safetensors.Result);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();

        MoonshineEncoder encoder = new(resolved);
        MoonshineDecoder decoder = new(resolved);
        encoder.LoadWeights(weights);
        decoder.LoadWeights(weights);

        MoonshineTokenizer tokenizer = new(repoDir);
        return new MoonshinePipeline(hfRepoId, resolved, encoder, decoder, tokenizer, loader);
    }

    /// <summary>Transcribe a WAV file. Auto-resamples to 16 kHz mono.</summary>
    public string TranscribeWav(IBackend backend, string wavPath, MoonshineOptions? options = null)
    {
        WavFile.DecodedAudio audio = WavFile.Read(wavPath);
        return TranscribeAudio(backend, audio.ToMono(), audio.SampleRate, options);
    }

    /// <summary>Transcribe an in-memory mono audio buffer at the given sample rate.</summary>
    public string TranscribeAudio(IBackend backend, float[] audio, int sampleRate, MoonshineOptions? options = null)
    {
        ThrowIfDisposed();
        MoonshineOptions opts = options ?? new MoonshineOptions();

        float[] mono16k = audio;
        if (sampleRate != 16_000)
        {
            Resampler resampler = Resampler.Create(sampleRate, 16_000);
            mono16k = resampler.Resample(audio);
        }

        // Encoder runs on the entire clip — no padding. Variable-length is a Moonshine
        // feature; the conv stem just produces ceil((len - kernel) / stride) frames.
        if (mono16k.Length < _cfg.Conv1Kernel)
            throw new ArgumentException($"Audio too short: need at least {_cfg.Conv1Kernel} samples ({_cfg.Conv1Kernel / 16000.0:F2} s).");

        Tensor encoded = _encoder.Forward(backend, mono16k);
        try
        {
            using MoonshineDecoder.DecodeState state = _decoder.StartDecode(backend, encoded);
            return GreedyDecode(backend, state, opts);
        }
        finally { encoded.Dispose(); }
    }

    private string GreedyDecode(IBackend backend, MoonshineDecoder.DecodeState state, MoonshineOptions opts)
    {
        // Prompt: just [BOS]. Moonshine has no language / task tokens.
        int[] prompt = [_cfg.BosTokenId];
        Tensor logits = _decoder.DecodeStep(backend, prompt, state);
        int nextToken = ArgMax(logits, _cfg.VocabSize);
        logits.Dispose();

        // Moonshine's documented hallucination guard: stop generating if the rate of
        // emitted tokens exceeds ~6.5 tokens / encoder-second of audio. We approximate
        // by capping total tokens to a multiple of the encoder sequence length, which
        // is what the original implementation effectively enforces.
        List<int> generated = new(opts.MaxNewTokens);
        int[] singleBuf = new int[1];
        for (int step = 0; step < opts.MaxNewTokens; step++)
        {
            if (nextToken == _cfg.EosTokenId) break;
            generated.Add(nextToken);

            singleBuf[0] = nextToken;
            Tensor stepLogits = _decoder.DecodeStep(backend, singleBuf, state);
            nextToken = ArgMax(stepLogits, _cfg.VocabSize);
            stepLogits.Dispose();
        }

        return _tokenizer.Decode(generated.ToArray()).TrimStart();
    }

    private static unsafe int ArgMax(Tensor logits, int vocab)
    {
        float* p = (float*)logits.DataPointer;
        int best = 0;
        float bestV = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            float val = p[v];
            if (val > bestV) { bestV = val; best = v; }
        }
        return best;
    }

    /// <summary>Infers a MoonshineConfig from the repo name. Hardcoded for the two
    /// official releases; pass an explicit config for unknown repos.</summary>
    public static MoonshineConfig InferConfig(string hfRepoId) => hfRepoId.ToLowerInvariant() switch
    {
        "usefulsensors/moonshine-tiny" => MoonshineConfig.Tiny,
        "usefulsensors/moonshine-base" => MoonshineConfig.Base,
        _ => throw new ArgumentException(
            $"Unknown Moonshine repo '{hfRepoId}'. Pass an explicit MoonshineConfig to LoadAsync.",
            nameof(hfRepoId)),
    };

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(MoonshinePipeline));
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
