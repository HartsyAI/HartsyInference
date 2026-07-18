using HartsyInference.Core.Backends;
using HartsyInference.Engine.Dispatch;

namespace HartsyInference.Engine;

/// <summary>The single in-process entry point for running inference: owns the compute backend, a cache of loaded
/// per-model runners, and the modality dispatch. A consumer (CLI, HTTP API, SwarmUI, direct library use) constructs
/// one, calls <see cref="Generate"/>, and disposes it — every generation across every modality flows through here.
/// Progress and artifact persistence stay with the caller (via the <see cref="IProgressSink"/> it passes in and the
/// returned <see cref="GeneratedArtifact"/>).</summary>
public sealed class InferenceEngine : IDisposable
{
    private readonly ModalityDispatch _dispatch = new ModalityDispatch();
    private readonly Dictionary<string, IModalityRunner> _runners = new(StringComparer.OrdinalIgnoreCase);
    private string _backendSelector;
    private IBackend? _backend;

    /// <summary>Creates an engine that lazily constructs the backend named by <paramref name="backendSelector"/>
    /// (<c>auto</c>/<c>cpu</c>/<c>cuda</c>/<c>vulkan</c>) on first use.</summary>
    public InferenceEngine(string backendSelector = "auto") => _backendSelector = backendSelector;

    /// <summary>The active backend selector.</summary>
    public string BackendSelector => _backendSelector;

    /// <summary>Human-readable description of what the selector resolves to (e.g. "auto → cuda").</summary>
    public string BackendDescription => BackendFactory.Describe(_backendSelector);

    /// <summary>Whether a handler is wired for <paramref name="modality"/>.</summary>
    public bool IsSupported(Modality modality) => _dispatch.IsSupported(modality);

    /// <summary>Loads (or reuses a cached) runner for <paramref name="spec"/> and generates. <paramref name="prompt"/>
    /// is the modality's primary input (text prompt, or an input file path for vision/3d/transcribe), and
    /// <paramref name="parameters"/> carries the per-modality tunables.</summary>
    public GeneratedArtifact Generate(ModelSpec spec, string prompt, ParamState parameters, IProgressSink progress, CancellationToken ct)
    {
        IModalityHandler handler = _dispatch.Get(spec.Modality);
        IModalityRunner runner = GetOrLoadRunner(spec, handler, progress);
        return handler.Run(runner, prompt, parameters, progress, ct);
    }

    /// <summary>Pre-loads the runner for <paramref name="spec"/> without generating (so a caller can print a header
    /// between load and run). Returns the same cached instance <see cref="Generate"/> will use.</summary>
    public IModalityRunner Load(ModelSpec spec, IProgressSink progress) =>
        GetOrLoadRunner(spec, _dispatch.Get(spec.Modality), progress);

    /// <summary>Switches the compute backend, disposing every loaded runner and the current backend (they are bound to
    /// the old device). Subsequent generations reload against the new backend.</summary>
    public void SetBackend(string selector)
    {
        _backendSelector = selector;
        DisposeLoaded();
    }

    private IModalityRunner GetOrLoadRunner(ModelSpec spec, IModalityHandler handler, IProgressSink progress)
    {
        string key = $"{spec.Modality}:{spec.LocalPath ?? spec.Requested}";
        if (_runners.TryGetValue(key, out IModalityRunner? cached))
            return cached;
        IModalityRunner runner = handler.Load(spec, EnsureBackend(), progress);
        _runners[key] = runner;
        return runner;
    }

    private IBackend EnsureBackend() => _backend ??= BackendFactory.Create(_backendSelector);

    private void DisposeLoaded()
    {
        foreach (IModalityRunner runner in _runners.Values)
            runner.Dispose();
        _runners.Clear();
        _backend?.Dispose();
        _backend = null;
    }

    /// <inheritdoc/>
    public void Dispose() => DisposeLoaded();
}
