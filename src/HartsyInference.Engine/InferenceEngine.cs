using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Registry;

namespace HartsyInference.Engine;

/// <summary>The single in-process entry point for running inference: owns the compute backend, the caches of loaded
/// pipelines, and the typed per-capability services. A consumer (CLI, HTTP API, SwarmUI, direct library use)
/// constructs one, calls the service it needs (<see cref="Images"/>, <see cref="Text"/>, …), and disposes it.
/// Progress flows through each service's <c>IProgress</c>/stream; results are returned as typed records.</summary>
public sealed class InferenceEngine : IInferenceEngine
{
    /// <summary>How long a release waits for an in-flight audio generation before dropping its pipelines anyway.</summary>
    private const int AudioUnloadWaitSeconds = 120;

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
    public bool IsSupported(Modality modality) => Modalities.All.Contains(modality);

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

    /// <inheritdoc/>
    public void SetBackend(string selector)
    {
        _backendSelector = selector;
        DisposeLoaded();
    }

    /// <summary>The compute backend, constructed on first use. Used by the typed services that drive pipelines directly
    /// (the recipe registry) rather than through a modality handler.</summary>
    internal IBackend Backend => EnsureBackend();

    /// <summary>Detects the checkpoint architecture for <paramref name="spec"/>, resolves its recipe, and constructs
    /// (or returns a cached) pipeline. Throws when no recipe is registered for the detected family yet.</summary>
    internal IRecipePipeline GetOrConstructRecipe(ModelSpec spec, ImageRequest? request = null)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                "No checkpoint found for this model. Pass a checkpoint via --model-path or let the catalog fetch it first.");
        }

        // LoRA and component overrides are baked into the loaded weights, so they are part of the cache identity.
        string key = $"recipe:{spec.LocalPath}|{RecipeCacheKey.Describe(request)}";
        if (_recipePipelines.TryGetValue(key, out IRecipePipeline? cached))
            return cached;

        IArchitectureRecipe recipe = ResolveRecipe(spec);
        IRecipePipeline pipeline = recipe.Construct(new RecipeContext
        {
            CheckpointPath = spec.LocalPath,
            Backend = EnsureBackend(),
            Components = request?.Components,
            Loras = request?.Loras,
        });
        _recipePipelines[key] = pipeline;
        return pipeline;
    }

    /// <summary>The composition features the recipe for <paramref name="spec"/> declares it can apply. Resolved through
    /// the same family-id + registry lookup <see cref="GetOrConstructRecipe"/> uses, so the answer can never disagree
    /// with the pipeline that will actually run.</summary>
    internal ImageFeatures SupportedFeatures(ModelSpec spec) => ResolveRecipe(spec).Supports;

    /// <summary>The officially recommended defaults for <paramref name="spec"/>: the constructed pipeline's
    /// variant-resolved numbers when it declares them, else the recipe's family-level ones. Resolved through the same
    /// family-id + registry lookup <see cref="GetOrConstructRecipe"/> uses, so the answer can never disagree with the
    /// pipeline that will actually run.</summary>
    internal ImageDefaults DefaultsFor(ModelSpec spec, IRecipePipeline pipeline)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        return pipeline.VariantDefaults ?? ResolveRecipe(spec).Defaults;
    }

    /// <summary>The officially recommended video defaults for <paramref name="spec"/>, resolved through the same
    /// family-id + registry lookup <see cref="GetOrConstructVideoRecipe"/> uses.</summary>
    internal static VideoDefaults VideoDefaultsFor(ModelSpec spec)
        => VideoRecipeRegistry.Resolve(ResolveFamilyId(spec))?.Defaults ?? VideoDefaults.Standard;

    /// <summary>The family id (catalog slug) that <paramref name="spec"/> resolves to, for diagnostics.</summary>
    internal static string FamilyIdFor(ModelSpec spec) => ResolveFamilyId(spec);

    /// <summary>Resolves the registered recipe for <paramref name="spec"/>, or throws naming what is drivable.</summary>
    private static IArchitectureRecipe ResolveRecipe(ModelSpec spec)
    {
        string familyId = ResolveFamilyId(spec);
        return RecipeRegistry.Resolve(familyId)
            ?? throw new NotSupportedException(
                $"Model family '{familyId}' has no recipe lifted into the Engine yet (E-IMG-3). " +
                $"Currently drivable: {string.Join(", ", RecipeRegistry.RegisteredNames)}.");
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

    private IBackend EnsureBackend() => _backend ??= BackendFactory.Create(_backendSelector);

    /// <inheritdoc/>
    public void FreeMemory() => ReleaseLoaded(disposeBackend: false);

    private void DisposeLoaded() => ReleaseLoaded(disposeBackend: true);

    /// <summary>Drops every loaded model across all modalities. With <paramref name="disposeBackend"/> the device goes
    /// too (teardown / backend switch); without it the backend survives and just has its device memory released, which
    /// is the "free memory" a host asks for between jobs.</summary>
    private void ReleaseLoaded(bool disposeBackend)
    {
        foreach (IRecipePipeline pipeline in _recipePipelines.Values)
            pipeline.Dispose();
        _recipePipelines.Clear();
        foreach (IVideoRecipePipeline pipeline in _videoRecipePipelines.Values)
            pipeline.Dispose();
        _videoRecipePipelines.Clear();
        // IsValueCreated throughout, so releasing memory never forces a service (and its caches) into existence.
        if (_vision.IsValueCreated)
        {
            _vision.Value.Dispose();
        }
        if (_mesh.IsValueCreated)
        {
            _mesh.Value.Dispose();
        }
        if (_world.IsValueCreated)
        {
            _world.Value.Dispose();
        }
        // TextService owns its own per-device backends and multi-GB dequantized host buffers, so it must be released
        // explicitly — nothing else here reaches its slots.
        if (_text.IsValueCreated)
        {
            _text.Value.Dispose();
        }
        // Audio pipelines are cached process-wide by the audio catalogs; drop them here so none outlives the backend
        // it was constructed against.
        Audio.AudioRuntime.UnloadAll(AudioUnloadWaitSeconds);
        if (disposeBackend)
        {
            _backend?.Dispose();
            _backend = null;
            return;
        }
        // Disposal only drops host references; the promoted GPU copies are freed on the finalizer queue, so force it
        // before asking the driver for the memory back or the card stays full.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        try
        {
            _backend?.FreeAllDeviceMemory();
            _backend?.TrimMemoryPool();
        }
        catch (Exception ex)
        {
            Logs.Warning($"[Engine] Releasing device memory failed: {ex.Message}");
        }
    }

    /// <inheritdoc/>
    public void Dispose() => DisposeLoaded();
}
