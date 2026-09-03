using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Audio.Models.Wake;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.Onnx;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Audio.Wake;

/// <summary>The wake models, loaded once and shared by every device.
///
/// <para>The front-end, backbone and heads are all stateless, and only the single wake worker thread ever calls
/// them, so one copy serves every satellite. Per-device state — the audio, mel and feature rings, and each
/// word's score smoothing — lives in that device's <see cref="WakeDetectionPipeline"/>.</para></summary>
public sealed class WakeModelSet : IDisposable
{
    private readonly Dictionary<string, (WakeHead Head, WakeWordConfig Config)> _heads = [];
    // Heads replaced by a reload. Live pipelines still borrow the old instance until the worker swaps them,
    // so disposing on replace would free tensors the worker is about to read.
    private readonly List<WakeHead> _retired = [];
    private readonly object _lock = new();
    private WakeMelFrontend? _mel;
    private SpeechEmbeddingModel? _embedding;
    private RnnoiseWeights? _denoiseWeights;
    private int _disposed;

    /// <summary>The wake assets directory this set was loaded from.</summary>
    public string ModelRoot { get; }

    public WakeModelSet(string modelRoot) => ModelRoot = modelRoot;

    /// <summary>Words currently loaded.</summary>
    public IReadOnlyCollection<string> Words
    {
        get { lock (_lock) return [.. _heads.Keys]; }
    }

    /// <summary>Loads the shared front-end and backbone, then every configured head. When <paramref name="words"/> is empty, every head found on disk is loaded with default settings.</summary>
    public void Load(IReadOnlyDictionary<string, WakeWordConfig> words)
    {
        string backbone = Path.Combine(ModelRoot, "backbone");
        string melPath = Path.Combine(backbone, "melspectrogram.onnx");
        string embeddingPath = Path.Combine(backbone, "embedding_model.onnx");
        if (!File.Exists(melPath) || !File.Exists(embeddingPath))
            throw new HartsyInferenceException($"Wake backbone not found under '{backbone}'. Expected melspectrogram.onnx and embedding_model.onnx.");

        using (OnnxWeightLoader loader = new())
        {
            loader.Load(melPath);
            WakeMelFrontend mel = new();
            mel.LoadWeights(loader.GetAllTensors());
            _mel = mel;
        }
        using (OnnxWeightLoader loader = new())
        {
            loader.Load(embeddingPath);
            SpeechEmbeddingModel embedding = new();
            embedding.LoadWeights(loader.GetAllTensors());
            _embedding = embedding;
        }

        string headDir = Path.Combine(ModelRoot, "heads");
        if (!Directory.Exists(headDir))
            throw new HartsyInferenceException($"Wake head directory '{headDir}' does not exist.");

        // Every head on disk is loaded; configuration adjusts a word's settings, it does not decide which words
        // exist. Treating the config file as an allowlist meant that setting one word's threshold silently
        // stopped every unconfigured word from loading at all.
        HashSet<string> loaded = new(StringComparer.OrdinalIgnoreCase);
        foreach (string file in Directory.EnumerateFiles(headDir, "*.onnx").Concat(Directory.EnumerateFiles(headDir, "*.safetensors")))
        {
            string name = Path.GetFileNameWithoutExtension(file);
            if (TryLoadHead(name, words.TryGetValue(name, out WakeWordConfig? existing) ? existing : new WakeWordConfig()))
            {
                loaded.Add(name);
            }
        }

        // A configured word may name a head file that differs from the word itself, so pick up anything the
        // directory sweep did not already cover.
        foreach ((string name, WakeWordConfig config) in words)
        {
            if (!loaded.Contains(name)) TryLoadHead(name, config);
        }

        if (_heads.Count == 0)
            throw new HartsyInferenceException($"No wake heads loaded from '{headDir}'; the listener would accept audio and never detect anything.");
        Logs.Info($"[Audio][Wake] Loaded {_heads.Count} wake word(s): {string.Join(", ", Words)}.");
    }

    /// <summary>Loads one head, adding it to any live pipelines the caller then refreshes. Returns false and logs when the head file is absent, so one bad entry does not prevent the rest from serving.</summary>
    public bool TryLoadHead(string name, WakeWordConfig config)
    {
        string stem = config.Head ?? name;
        string headDir = Path.Combine(ModelRoot, "heads");
        string path = new[] { ".onnx", ".safetensors" }
            .Select(ext => Path.Combine(headDir, stem + ext))
            .FirstOrDefault(File.Exists) ?? "";
        if (path.Length == 0)
        {
            Logs.Warning($"[Audio][Wake] Wake word '{name}' has no head file '{stem}.onnx' or '{stem}.safetensors' under '{headDir}'; skipping it.");
            return false;
        }

        try
        {
            WakeHead head = new(name);
            // Shipped heads are ONNX; heads trained in-engine are safetensors. Both are weights-only reads.
            if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
            {
                using SafeTensorsLoader loader = new();
                loader.Load(path);
                head.LoadWeights(loader.GetAllTensors());
            }
            else
            {
                using OnnxWeightLoader loader = new();
                loader.Load(path);
                head.LoadWeights(loader.GetAllTensors());
            }
            lock (_lock)
            {
                if (_heads.Remove(name, out (WakeHead Head, WakeWordConfig Config) old)) _retired.Add(old.Head);
                _heads[name] = (head, config);
            }
            return true;
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio][Wake] Failed to load wake word '{name}' from '{path}'.", ex);
            return false;
        }
    }

    /// <summary>Whether noise suppression is available, i.e. <see cref="LoadDenoiser"/> found weights.</summary>
    public bool DenoiseAvailable => _denoiseWeights?.IsLoaded == true;

    /// <summary>Loads the RNNoise weights from <c>{ModelRoot}/denoise</c>. Returns false and logs when they are
    /// absent — a missing optional model must not stop the listener from hearing anything, so the caller runs
    /// without suppression rather than failing to start.</summary>
    public bool LoadDenoiser()
    {
        string path = Path.Combine(ModelRoot, "denoise", "rnnoise.safetensors");
        if (!File.Exists(path))
        {
            Logs.Warning($"[Audio][Wake] Noise suppression is enabled but no denoiser was found at '{path}'. Listening without it.");
            return false;
        }
        try
        {
            RnnoiseWeights weights = new();
            using (SafeTensorsLoader loader = new())
            {
                loader.Load(path);
                weights.Load(loader.GetAllTensors());
            }
            _denoiseWeights = weights;
            Logs.Info($"[Audio][Wake] Noise suppression enabled (weights from '{path}').");
            return true;
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio][Wake] Failed to load the denoiser from '{path}'; listening without it.", ex);
            return false;
        }
    }

    /// <summary>A per-device denoiser over the shared weights, or null when suppression is off or unavailable.
    /// The weights are shared; only the stream state is per-device.</summary>
    public RnnoiseStream? CreateDenoiser() =>
        _denoiseWeights is null ? null : new RnnoiseStream(_denoiseWeights, WakeAudioRate);

    /// <summary>The rate satellites stream at, and the rate the detection pipeline expects.</summary>
    private const int WakeAudioRate = 16_000;

    /// <summary>Builds a fresh per-device pipeline wired to the shared models and every loaded word.</summary>
    public WakeDetectionPipeline CreatePipeline()
    {
        if (_mel is null || _embedding is null) throw new InvalidOperationException("WakeModelSet.Load was not called.");
        WakeDetectionPipeline pipeline = new(_mel, _embedding);
        lock (_lock)
        {
            foreach ((WakeHead head, WakeWordConfig config) in _heads.Values)
                pipeline.AddWord(head, ToSettings(config));
        }
        return pipeline;
    }

    /// <summary>The head and settings for one word, for a caller applying it to a live pipeline.</summary>
    public (WakeHead Head, WakeWordConfig Config)? Entry(string word)
    {
        lock (_lock) return _heads.TryGetValue(word, out (WakeHead Head, WakeWordConfig Config) entry) ? entry : null;
    }

    /// <summary>Settings converted for the pipeline.</summary>
    public static WakeWordSettings Settings(WakeWordConfig config) => ToSettings(config);

    /// <summary>The configuration a word fired under, for annotating its detection event.</summary>
    public WakeWordConfig? ConfigFor(string word)
    {
        lock (_lock) return _heads.TryGetValue(word, out (WakeHead Head, WakeWordConfig Config) entry) ? entry.Config : null;
    }

    private static WakeWordSettings ToSettings(WakeWordConfig config) => new()
    {
        Threshold = config.Threshold,
        SmoothingWindow = config.SmoothingWindow,
        RefractorySeconds = config.RefractorySeconds,
    };

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        lock (_lock)
        {
            foreach ((WakeHead head, _) in _heads.Values) head.Dispose();
            _heads.Clear();
            foreach (WakeHead head in _retired) head.Dispose();
            _retired.Clear();
        }
        _mel?.Dispose();
        _embedding?.Dispose();
        // Safe only because every RnnoiseStream borrowing these belongs to a session, and sessions are disposed
        // before the model set is.
        _denoiseWeights?.Dispose();
    }
}
