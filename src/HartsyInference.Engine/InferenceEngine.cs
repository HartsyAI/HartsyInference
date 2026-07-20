using HartsyInference.Core.Backends;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Services;
using HartsyInference.ModelHandler.Registry;

namespace HartsyInference.Engine;

/// <summary>The single in-process entry point for running inference: owns the compute backend, a cache of loaded
/// per-model runners, the modality dispatch, and the typed per-capability services. A consumer (CLI, HTTP API,
/// SwarmUI, direct library use) constructs one, calls the service it needs (<see cref="Images"/>, <see cref="Text"/>,
/// …), and disposes it. Progress flows through each service's <c>IProgress</c>/stream; results are returned as typed
/// records.</summary>
public sealed class InferenceEngine : IInferenceEngine
{
    private readonly ModalityDispatch _dispatch = new ModalityDispatch();
    private readonly Dictionary<string, IModalityRunner> _runners = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, IRecipePipeline> _recipePipelines = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, IVideoRecipePipeline> _videoRecipePipelines = new(StringComparer.OrdinalIgnoreCase);
    private string _backendSelector;
    private IBackend? _backend;

    private readonly Lazy<ImagesService> _images;
    private readonly Lazy<VideoService> _video;
    private readonly Lazy<TextService> _text;
    private readonly Lazy<MusicService> _music;
    private readonly Lazy<SpeechService> _speech;
    private readonly Lazy<TranscribeService> _transcribe;
    private readonly Lazy<VoiceConversionService> _voiceConversion;
    private readonly Lazy<FxService> _fx;
    private readonly Lazy<VisionService> _vision;
    private readonly Lazy<MeshService> _mesh;
    private readonly Lazy<WorldService> _world;

    /// <summary>Creates an engine that lazily constructs the backend named by <paramref name="backendSelector"/>
    /// (<c>auto</c>/<c>cpu</c>/<c>cuda</c>/<c>vulkan</c>) on first use.</summary>
    public InferenceEngine(string backendSelector = "auto")
    {
        _backendSelector = backendSelector;
        _images = new Lazy<ImagesService>(() => new ImagesService(this));
        _video = new Lazy<VideoService>(() => new VideoService(this));
        _text = new Lazy<TextService>(() => new TextService(this));
        _music = new Lazy<MusicService>(() => new MusicService(this));
        _speech = new Lazy<SpeechService>(() => new SpeechService(this));
        _transcribe = new Lazy<TranscribeService>(() => new TranscribeService(this));
        _voiceConversion = new Lazy<VoiceConversionService>(() => new VoiceConversionService(this));
        _fx = new Lazy<FxService>(() => new FxService(this));
        _vision = new Lazy<VisionService>(() => new VisionService(this));
        _mesh = new Lazy<MeshService>(() => new MeshService(this));
        _world = new Lazy<WorldService>(() => new WorldService(this));
    }

    /// <inheritdoc/>
    public string BackendSelector => _backendSelector;

    /// <inheritdoc/>
    public string BackendDescription => BackendFactory.Describe(_backendSelector);

    /// <inheritdoc/>
    public bool IsSupported(Modality modality) => _dispatch.IsSupported(modality);

    /// <inheritdoc/>
    public IImagesService Images => _images.Value;

    /// <inheritdoc/>
    public IVideoService Video => _video.Value;

    /// <inheritdoc/>
    public ITextService Text => _text.Value;

    /// <inheritdoc/>
    public IMusicService Music => _music.Value;

    /// <inheritdoc/>
    public ISpeechService Speech => _speech.Value;

    /// <inheritdoc/>
    public ITranscribeService Transcribe => _transcribe.Value;

    /// <inheritdoc/>
    public IVoiceConversionService VoiceConversion => _voiceConversion.Value;

    /// <inheritdoc/>
    public IFxService Fx => _fx.Value;

    /// <inheritdoc/>
    public IVisionService Vision => _vision.Value;

    /// <inheritdoc/>
    public IMeshService Mesh => _mesh.Value;

    /// <inheritdoc/>
    public IWorldService World => _world.Value;

    /// <summary>Pre-loads the runner for <paramref name="spec"/> without generating (so a caller can print a header
    /// between load and run). Returns the same cached instance the services will use.</summary>
    public IModalityRunner Load(ModelSpec spec, IProgressSink progress) =>
        GetOrLoadRunner(spec, _dispatch.Get(spec.Modality), progress);

    /// <inheritdoc/>
    public void SetBackend(string selector)
    {
        _backendSelector = selector;
        DisposeLoaded();
    }

    /// <summary>Transitional generic generation entry used by the CLI until every modality has a complete typed
    /// service. New consumers (the SwarmUI extension, library users) should use the typed services
    /// (<see cref="Images"/>, <see cref="Text"/>, …); this is removed once the CLI is rewired onto them.</summary>
    public GeneratedArtifact Generate(ModelSpec spec, string prompt, ParamState parameters, IProgressSink progress, CancellationToken ct) =>
        RunHandler(spec, prompt, parameters, progress, ct);

    /// <summary>Loads (or reuses a cached) runner for <paramref name="spec"/> and runs its handler. The internal
    /// bridge the typed services drive; every generation flows through here.</summary>
    internal GeneratedArtifact RunHandler(ModelSpec spec, string prompt, ParamState parameters, IProgressSink progress, CancellationToken ct)
    {
        IModalityHandler handler = _dispatch.Get(spec.Modality);
        IModalityRunner runner = GetOrLoadRunner(spec, handler, progress);
        return handler.Run(runner, prompt, parameters, progress, ct);
    }

    /// <summary>The compute backend, constructed on first use. Used by the typed services that drive pipelines directly
    /// (the recipe registry) rather than through a modality handler.</summary>
    internal IBackend Backend => EnsureBackend();

    /// <summary>Detects the checkpoint architecture for <paramref name="spec"/>, resolves its recipe, and constructs
    /// (or returns a cached) pipeline. Throws when no recipe is registered for the detected family yet.</summary>
    internal IRecipePipeline GetOrConstructRecipe(ModelSpec spec)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                "No checkpoint found for this model. Pass a checkpoint via --model-path or let the catalog fetch it first.");
        }

        string key = $"recipe:{spec.LocalPath}";
        if (_recipePipelines.TryGetValue(key, out IRecipePipeline? cached))
            return cached;

        string familyId = ResolveFamilyId(spec);
        IArchitectureRecipe recipe = RecipeRegistry.Resolve(familyId)
            ?? throw new NotSupportedException(
                $"Model family '{familyId}' has no recipe lifted into the Engine yet (E-IMG-3). " +
                $"Currently drivable: {string.Join(", ", RecipeRegistry.RegisteredNames)}.");

        IRecipePipeline pipeline = recipe.Construct(new RecipeContext { CheckpointPath = spec.LocalPath, Backend = EnsureBackend() });
        _recipePipelines[key] = pipeline;
        return pipeline;
    }

    /// <summary>Resolves the video recipe for <paramref name="spec"/> and constructs (or returns a cached) pipeline.
    /// Throws when no video recipe is registered for the family yet.</summary>
    internal IVideoRecipePipeline GetOrConstructVideoRecipe(ModelSpec spec)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                "No checkpoint found for this model. Pass a checkpoint via --model-path or let the catalog fetch it first.");
        }

        string key = $"video-recipe:{spec.LocalPath}";
        if (_videoRecipePipelines.TryGetValue(key, out IVideoRecipePipeline? cached))
            return cached;

        string familyId = ResolveFamilyId(spec);
        IVideoRecipe recipe = VideoRecipeRegistry.Resolve(familyId)
            ?? throw new NotSupportedException(
                $"Video family '{familyId}' has no recipe lifted into the Engine yet (E-IMG-3). " +
                $"Currently drivable: {string.Join(", ", VideoRecipeRegistry.RegisteredNames)}.");

        IVideoRecipePipeline pipeline = recipe.Construct(new RecipeContext { CheckpointPath = spec.LocalPath, Backend = EnsureBackend() });
        _videoRecipePipelines[key] = pipeline;
        return pipeline;
    }

    /// <summary>The family id (catalog slug) for <paramref name="spec"/>: the catalog id when present, else a slug
    /// mapped from the coarse tensor-signature architecture the Engine can detect from a raw checkpoint.</summary>
    private static string ResolveFamilyId(ModelSpec spec)
    {
        if (spec.Catalog is not null)
            return spec.Catalog.Id;
        ModelArchitecture arch = PipelineFactory.DetectArchitecture(spec.LocalPath!);
        return arch switch
        {
            ModelArchitecture.Sdxl => "sdxl",
            ModelArchitecture.SdxlRefiner => "sdxl-refiner",
            ModelArchitecture.StableDiffusion15 => "sd15",
            ModelArchitecture.StableDiffusion3 => "sd3",
            ModelArchitecture.Flux1 => "flux1",
            ModelArchitecture.Flux2 => "flux2",
            ModelArchitecture.AuraFlow => "auraflow",
            ModelArchitecture.Chroma => "chroma",
            _ => arch.ToString().ToLowerInvariant(),
        };
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
        foreach (IRecipePipeline pipeline in _recipePipelines.Values)
            pipeline.Dispose();
        _recipePipelines.Clear();
        foreach (IVideoRecipePipeline pipeline in _videoRecipePipelines.Values)
            pipeline.Dispose();
        _videoRecipePipelines.Clear();
        _backend?.Dispose();
        _backend = null;
    }

    /// <inheritdoc/>
    public void Dispose() => DisposeLoaded();
}
