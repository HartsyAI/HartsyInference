using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Audio.Models.Wake;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Tensors;
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
    private string? _vadWeightsPath;
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

    /// <summary>Whether an end-of-speech detector is available, i.e. <see cref="LoadVad"/> found weights.</summary>
    public bool VadAvailable => _vadWeightsPath is not null;

    /// <summary>Locates the Silero VAD weights under <c>{ModelRoot}/vad</c>.
    ///
    /// <para>Optional, like the denoiser: without it the service falls back to a fixed post-detection wait,
    /// which works but cuts off anyone who asks a long question. A missing file must not stop the listener
    /// from hearing anything, so this logs and returns false rather than throwing.</para>
    ///
    /// <para>Only the path is checked here. Unlike the denoiser's weights, which are shared across devices,
    /// each device needs its own <see cref="SileroVad"/>: the model carries LSTM state for one stream, so a
    /// shared instance would have two people's speech interleaved into one hidden vector.</para></summary>
    public bool LoadVad()
    {
        string dir = Path.Combine(ModelRoot, "vad");
        string safetensors = Path.Combine(dir, "silero_vad_16k.safetensors");
        string onnx = Path.Combine(dir, "silero_vad.onnx");
        string path = File.Exists(safetensors) ? safetensors : onnx;
        if (!File.Exists(path))
        {
            Logs.Warning($"[Audio][Wake] No end-of-speech model in '{dir}' (looked for silero_vad_16k.safetensors and silero_vad.onnx); falling back to a fixed post-detection wait, which truncates long questions. Install it from the Wake Word tab.");
            return false;
        }
        try
        {
            // Load once here to fail loudly at startup rather than on the first person who speaks.
            using (SileroVad probe = LoadVadFrom(path)) { }
            _vadWeightsPath = path;
            Logs.Info($"[Audio][Wake] End-of-speech detection enabled (weights from '{path}').");
            return true;
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio][Wake] Failed to load the end-of-speech model from '{path}'; using the fixed wait instead.", ex);
            return false;
        }
    }

    private static SileroVad LoadVadFrom(string path)
    {
        SileroVad vad = new();
        try
        {
            // The loader stays alive across LoadWeights: the tensors it hands out are its own, and
            // WakeWeights.Own copies them into the model precisely because the loader is transient.
            if (path.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            {
                using OnnxWeightLoader onnx = new();
                onnx.Load(path);
                vad.LoadWeights(SileroTensorsFromOnnx(onnx, path));
            }
            else
            {
                using SafeTensorsLoader safe = new();
                safe.Load(path);
                vad.LoadWeights(safe.GetAllTensors());
            }
            return vad;
        }
        catch
        {
            vad.Dispose();
            throw;
        }
    }

    /// <summary>The fifteen weights SileroVad needs, in the order their shapes appear in the ONNX's 16 kHz branch.</summary>
    /// <remarks>Shapes are from docs/Research/WAKE_WORD_DETECTION.md and double as the check that this really is
    /// Silero v6 rather than something else with the same file name.</remarks>
    private static readonly (string Name, int[] Shape)[] SileroLayout =
    [
        ("stft_conv.weight", [258, 1, 256]),
        ("conv1.weight", [128, 129, 3]), ("conv1.bias", [128]),
        ("conv2.weight", [64, 128, 3]), ("conv2.bias", [64]),
        ("conv3.weight", [64, 64, 3]), ("conv3.bias", [64]),
        ("conv4.weight", [128, 64, 3]), ("conv4.bias", [128]),
        ("lstm_cell.weight_ih", [512, 128]), ("lstm_cell.weight_hh", [512, 128]),
        ("lstm_cell.bias_ih", [512]), ("lstm_cell.bias_hh", [512]),
        ("final_conv.weight", [1, 128, 1]), ("final_conv.bias", [1]),
    ];

    /// <summary>Reads Silero's weights straight out of the upstream <c>silero_vad.onnx</c>.
    ///
    /// <para>This exists so the model can be installed from its canonical MIT source with no conversion step and
    /// nobody having to host a repacked copy. It is an awkward file to read: the network is wrapped in an
    /// <c>If</c> on sample rate, and the weights are anonymous <c>Constant</c> nodes inside each branch rather
    /// than graph initializers, so there are no names to bind by.</para>
    ///
    /// <para>They are recovered by shape from the 16 kHz branch. Thirteen of the fifteen shapes are unique, which
    /// pins those outright; the two <c>[512, 128]</c> LSTM matrices and the two <c>[512]</c> biases go in graph
    /// order, which is PyTorch's — input-hidden before hidden-hidden. That ordering is confirmed against
    /// onnxruntime by <c>tools/convert_silero_onnx.py --verify</c>, and getting it wrong does not fail quietly:
    /// the probabilities come out obviously wrong rather than subtly off.</para></summary>
    private static Dictionary<string, Tensor> SileroTensorsFromOnnx(OnnxWeightLoader loader, string path)
    {
        // then_branch is the 16 kHz path; else_branch holds the same architecture trained for 8 kHz.
        IReadOnlyList<Tensor> constants = loader.SubgraphConstants("then_branch");
        Dictionary<string, Tensor> weights = [];
        List<Tensor> pool = [.. constants];
        foreach ((string name, int[] shape) in SileroLayout)
        {
            int index = pool.FindIndex(t => MatchesShape(t, shape));
            if (index < 0)
            {
                throw new HartsyInferenceException(
                    $"'{path}' has no {string.Join("x", shape)} tensor left for '{name}'. This does not look " +
                    $"like Silero VAD v6 — source it from github.com/snakers4/silero-vad, not from an " +
                    $"openWakeWord release (which pins an older revision).");
            }
            weights[name] = pool[index];
            pool.RemoveAt(index);
        }
        return weights;
    }

    private static bool MatchesShape(Tensor tensor, int[] shape)
    {
        if (tensor.DType != DType.F32 || tensor.Shape.Rank != shape.Length) return false;
        for (int i = 0; i < shape.Length; i++)
        {
            if (tensor.Shape[i] != shape[i]) return false;
        }
        return true;
    }

    /// <summary>A per-device end-of-speech detector, or null when none is available. Each carries its own LSTM
    /// state, so one per stream.</summary>
    public SileroVadStream? CreateVad(int minSilenceMs)
    {
        if (_vadWeightsPath is null) return null;
        try
        {
            return new SileroVadStream(LoadVadFrom(_vadWeightsPath), minSilenceMs: minSilenceMs);
        }
        catch (Exception ex)
        {
            Logs.Error("[Audio][Wake] Could not create an end-of-speech detector for a device; it will use the fixed wait.", ex);
            return null;
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
