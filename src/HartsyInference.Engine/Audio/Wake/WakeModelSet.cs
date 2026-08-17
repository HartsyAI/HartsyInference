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
    private int _disposed;

    /// <summary>The wake assets directory this set was loaded from.</summary>
    public string ModelRoot { get; }

    public WakeModelSet(string modelRoot) => ModelRoot = modelRoot;

    /// <summary>Words currently loaded.</summary>
    public IReadOnlyCollection<string> Words
    {
        get { lock (_lock) return [.. _heads.Keys]; }
    }

    /// <summary>Loads the shared front-end and backbone, then every configured head. When
    /// <paramref name="words"/> is empty, every head found on disk is loaded with default settings.</summary>
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

        if (words.Count == 0)
        {
            foreach (string file in Directory.EnumerateFiles(headDir, "*.onnx").Concat(Directory.EnumerateFiles(headDir, "*.safetensors")))
            {
                string name = Path.GetFileNameWithoutExtension(file);
                TryLoadHead(name, new WakeWordConfig());
            }
        }
        else
        {
            foreach ((string name, WakeWordConfig config) in words)
                TryLoadHead(name, config);
        }

        if (_heads.Count == 0)
            throw new HartsyInferenceException($"No wake heads loaded from '{headDir}'; the listener would accept audio and never detect anything.");
        Logs.Info($"[Audio][Wake] Loaded {_heads.Count} wake word(s): {string.Join(", ", Words)}.");
    }

    /// <summary>Loads one head, adding it to any live pipelines the caller then refreshes. Returns false and logs
    /// when the head file is absent, so one bad entry does not prevent the rest from serving.</summary>
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
    }
}
