using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Moonshine;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>End-to-end 2nd-gen streaming Moonshine STT pipeline (<c>UsefulSensors/moonshine-streaming-
/// {tiny,small,medium}</c>): raw 16-kHz mono waveform → frame-CMVN + asinh-compressed causal-conv front
/// end → sliding-window-attention encoder (no RoPE) → RoPE cross-attention decoder (untied LM head,
/// SwiGLU MLP) → SentencePiece byte-fallback BPE decode → text. See <see cref="MoonshineStreamingEncoder"/>
/// / <see cref="MoonshineStreamingDecoder"/> for the architecture deltas from the original
/// <see cref="MoonshinePipeline"/>.
///
/// <para>This is a full-utterance batch pipeline (encode the whole clip, then autoregressively decode
/// with a KV cache) — bit-exact against the HF reference forward pass, which is itself batch-only. True
/// chunked/incremental audio ingestion (exploiting each encoder layer's bounded left/right sliding
/// window so the encoder never needs to re-run over already-seen audio) is NOT implemented; the encoder
/// re-processes the full clip per call, same as Whisper/the original Moonshine in this engine today.</para></summary>
public sealed class MoonshineStreamingPipeline : IAudioPipeline, IDisposable
{
    private readonly MoonshineConfig _cfg;
    private readonly MoonshineStreamingEncoder _encoder;
    private readonly MoonshineStreamingDecoder _decoder;
    private readonly MoonshineTokenizer _tokenizer;
    private readonly SafeTensorsLoader _loader;
    private int _disposed;

    public string ModelName { get; }

    private MoonshineStreamingPipeline(string modelName, MoonshineConfig cfg, MoonshineStreamingEncoder enc,
        MoonshineStreamingDecoder dec, MoonshineTokenizer tok, SafeTensorsLoader loader)
    {
        ModelName = modelName;
        _cfg = cfg;
        _encoder = enc;
        _decoder = dec;
        _tokenizer = tok;
        _loader = loader;
    }

    /// <summary>Loads a streaming Moonshine pipeline from a HuggingFace repo. Auto-downloads
    /// model.safetensors and tokenizer.json on first use.</summary>
    public static async Task<MoonshineStreamingPipeline> LoadAsync(string hfRepoId, MoonshineConfig? cfg = null, CancellationToken ct = default)
    {
        MoonshineConfig resolved = cfg ?? InferConfig(hfRepoId);
        string repoDir = AudioModelCache.GetRepoDirectory(hfRepoId, "stt");

        Task<string> safetensors = AudioModelCache.GetAsync(hfRepoId, "model.safetensors", category: "stt", ct: ct);
        Task<string> tokenizerJson = AudioModelCache.GetAsync(hfRepoId, "tokenizer.json", category: "stt", ct: ct);
        await Task.WhenAll(safetensors, tokenizerJson).ConfigureAwait(false);

        SafeTensorsLoader loader = new();
        loader.Load(safetensors.Result);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();

        MoonshineStreamingEncoder encoder = new(resolved);
        MoonshineStreamingDecoder decoder = new(resolved);
        encoder.LoadWeights(weights);
        decoder.LoadWeights(weights);

        MoonshineTokenizer tokenizer = new(repoDir);
        return new MoonshineStreamingPipeline(hfRepoId, resolved, encoder, decoder, tokenizer, loader);
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
        if (sampleRate != _cfg.SampleRate)
        {
            Resampler resampler = Resampler.Create(sampleRate, _cfg.SampleRate);
            mono16k = resampler.Resample(audio);
        }

        int frameLen = (int)Math.Round(_cfg.SampleRate * _cfg.FrameMs / 1000.0);
        if (mono16k.Length < frameLen)
            throw new ArgumentException($"Audio too short: need at least {frameLen} samples ({frameLen / (double)_cfg.SampleRate:F2} s).");
        // Trim to a whole number of frames — the embedder reshapes samples into fixed-length frames.
        int usable = (mono16k.Length / frameLen) * frameLen;
        if (usable != mono16k.Length) mono16k = mono16k[..usable];

        Tensor encoded = _encoder.Forward(backend, mono16k);
        try
        {
            using MoonshineDecoder.DecodeState state = _decoder.StartDecode(backend, encoded);
            return MoonshineGreedyDecoder.Decode(backend, _decoder.DecodeStep, state, _cfg, _tokenizer, opts);
        }
        finally { encoded.Dispose(); }
    }

    /// <summary>Infers a streaming <see cref="MoonshineConfig"/> preset from the repo name.</summary>
    public static MoonshineConfig InferConfig(string hfRepoId) => hfRepoId.ToLowerInvariant() switch
    {
        "usefulsensors/moonshine-streaming-tiny" => MoonshineConfig.StreamingTiny,
        "usefulsensors/moonshine-streaming-small" => MoonshineConfig.StreamingSmall,
        "usefulsensors/moonshine-streaming-medium" => MoonshineConfig.StreamingMedium,
        _ => throw new ArgumentException(
            $"Unknown streaming Moonshine repo '{hfRepoId}'. Pass an explicit MoonshineConfig to LoadAsync.",
            nameof(hfRepoId)),
    };

    private void ThrowIfDisposed()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(MoonshineStreamingPipeline));
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
