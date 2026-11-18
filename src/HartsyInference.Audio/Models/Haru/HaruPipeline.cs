using HartsyInference.Audio.Models.VibeVoice;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Models.Haru;

/// <summary>Haru-4B loaded and resident on a backend: the language model, the MeanFlow head and the codec, with the weights held at a
/// chosen <see cref="HaruPrecision"/>. It serves one conversation at a time (<see cref="OpenSession"/>), because the language model's
/// recurrent state and attention cache belong to the conversation.</summary>
public sealed class HaruPipeline : IDisposable
{
    /// <summary>Positions the model is loaded for unless the caller says otherwise: about 13 minutes of conversation and the prompt.</summary>
    public const int DefaultCapacity = 6000;

    private readonly HaruConfig _config;
    private readonly HaruLm _lm;
    private readonly HaruMeanFlowHead _head;
    private readonly HaruCodecSession _codec;
    private readonly VibeVoiceAcousticTokenizerModel _codecModel;
    private readonly HaruTextCodec _text;
    private readonly IBackend _backend;
    private readonly IDisposable _weightsOwner;
    private int _active;
    private int _disposed;
    private int _streamWarned;

    private HaruPipeline(HaruConfig config, HaruLm lm, HaruMeanFlowHead head, HaruCodecSession codec, VibeVoiceAcousticTokenizerModel codecModel,
        HaruTextCodec text, IBackend backend, IDisposable weightsOwner)
    {
        _config = config;
        _lm = lm;
        _head = head;
        _codec = codec;
        _codecModel = codecModel;
        _text = text;
        _backend = backend;
        _weightsOwner = weightsOwner;
    }

    /// <summary>Sample rate of both audio streams: 24 kHz.</summary>
    public int SampleRate => _config.SampleRate;

    /// <summary>Samples in one frame of either stream: 3200, 133 ms.</summary>
    public int FrameSamples => _config.FrameSamples;

    /// <summary>Positions the model is loaded for.</summary>
    public int Capacity => _lm.Capacity;

    /// <summary>Loads Haru-4B from the three files of the release. A <paramref name="precision"/> other than native converts the language
    /// model's matrices once into the cache at <paramref name="cachePath"/>. The weights are made resident on <paramref name="backend"/>
    /// where they fit.</summary>
    public static HaruPipeline Load(IBackend backend, string configPath, string tokenizerPath, string weightsPath, HaruPrecision precision,
        string cachePath, int capacity = DefaultCapacity)
    {
        ArgumentNullException.ThrowIfNull(backend);
        HaruConfig config = HaruConfig.FromJson(File.ReadAllText(configPath));
        HaruTextCodec text;
        using (FileStream json = File.OpenRead(tokenizerPath))
        {
            text = HaruTextCodec.From(HaruTokenizer.Load(json, config));
        }
        HfQuantCheckpoint checkpoint = HfQuantCheckpoint.OpenFiles([weightsPath]);
        IReadOnlyDictionary<string, Tensor> weights;
        IDisposable owner;
        try
        {
            (weights, owner) = HaruWeights.Load(checkpoint, precision, cachePath);
        }
        catch
        {
            checkpoint.Dispose();
            throw;
        }
        try
        {
            return Build(backend, config, text, weights, owner, capacity);
        }
        catch
        {
            owner.Dispose();
            throw;
        }
    }

    /// <summary>Builds the pipeline over already-mapped weights, which <paramref name="owner"/> keeps alive; the text codec stands in for a
    /// tokenizer file.</summary>
    internal static HaruPipeline Build(IBackend backend, HaruConfig config, HaruTextCodec text, IReadOnlyDictionary<string, Tensor> weights,
        IDisposable owner, int capacity)
    {
        HaruLm lm = new(config, weights, capacity);
        HaruMeanFlowHead? head = null;
        HaruCodecSession? codec = null;
        try
        {
            head = new HaruMeanFlowHead(config.Head);
            head.LoadWeights(weights);
            VibeVoiceAcousticTokenizerModel codecModel = HaruCodec.Create(config.Codec, weights, keepStoredWeight: backend.Device.IsCuda);
            codec = new HaruCodecSession(codecModel, backend, config.LatentDim, config.Codec.FixStd);
            HaruPipeline pipeline = new(config, lm, head, codec, codecModel, text, backend, owner);
            pipeline.Preload();
            Logs.Info($"[Audio][Haru] Loaded: {config.Lm.NumLayers}-layer Qwen3.5 trunk, {config.FrameRate:0.#} Hz frames, capacity {lm.Capacity} positions.");
            return pipeline;
        }
        catch
        {
            // The weights owner stays with the caller (Load disposes it); the parts made here do not outlive a failed build.
            codec?.Dispose();
            head?.Dispose();
            lm.Dispose();
            throw;
        }
    }

    /// <summary>A voice reference (mono, 24 kHz) as the model is prompted with it: silence trimmed, the last three minutes kept.</summary>
    public static float[] PrepareReference(ReadOnlySpan<float> pcm24k) => HaruAudio.PrepareReference(pcm24k, 24000);

    /// <summary>Starts a conversation. Only one may be open: dispose the session before opening the next.</summary>
    public HaruDuplexSession OpenSession(HaruDuplexOptions options) => OpenSession(options, null);

    internal HaruDuplexSession OpenSession(HaruDuplexOptions options, IHaruNoise? noise)
    {
        ArgumentNullException.ThrowIfNull(options);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (Interlocked.CompareExchange(ref _active, 1, 0) != 0)
        {
            throw new InvalidOperationException("A Haru conversation is already open on this model; dispose it first.");
        }
        try
        {
            return new HaruDuplexSession(_config, _lm, _head, _codec, _text, _backend, options, noise, () => Volatile.Write(ref _active, 0));
        }
        catch
        {
            Volatile.Write(ref _active, 0);
            throw;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        _codec.Dispose();
        _head.Dispose();
        _lm.Dispose();
        _weightsOwner.Dispose();
    }

    /// <summary>Keeps every weight resident where it fits; a card that cannot hold them all streams the rest per use.</summary>
    private void Preload()
    {
        List<Tensor> weights = [.. _lm.Trunk.EnumerateWeights(), .. _lm.EnumerateWeights(), .. _head.EnumerateWeights(), .. _codecModel.EnumerateWeights()];
        WhisperOps.PreloadOrStream(_backend, [.. weights], "Haru", ref _streamWarned);
    }
}
