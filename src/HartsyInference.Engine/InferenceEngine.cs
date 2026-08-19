using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Memory;
using HartsyInference.Core.MemoryManagement;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Placement;
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
    private readonly EngineOptions? _options;
    private Audio.AudioRuntime? _audioRuntime;
    private PlacementConfig _placement = PlacementConfig.Single;

    /// <summary>Placement-resolved SECONDARY backends (text-encoder device, shard stages), keyed by canonical
    /// selector. The primary stays in <see cref="_backend"/>; this pool only fills when a placement names other
    /// devices, so single-device engines never pay for it.</summary>
    private readonly Dictionary<string, IBackend> _placementBackends = new(StringComparer.OrdinalIgnoreCase);

    private readonly Lazy<ImagesService> _images;
    private readonly Lazy<VideoService> _video;
    private readonly Lazy<TextService> _text;
    private readonly Lazy<MusicService> _music;
    private readonly Lazy<SpeechService> _speech;
    private readonly Lazy<TranscribeService> _transcribe;
    private readonly Lazy<VoiceConversionService> _voiceConversion;
    private readonly Lazy<FxService> _fx;
    private readonly Lazy<VisionService> _vision;
    private readonly Lazy<RestoreService> _restore;
    private readonly Lazy<MeshService> _mesh;
    private readonly Lazy<WorldService> _world;
    private readonly Lazy<EmbeddingService> _embeddings;

    /// <summary>Creates an engine that lazily constructs the backend named by <paramref name="backendSelector"/>
    /// (<c>auto</c>/<c>cpu</c>/<c>cuda</c>/<c>vulkan</c>, optionally with a <c>:{ordinal}</c> device suffix) on first use.</summary>
    public InferenceEngine(string backendSelector = "auto", EngineOptions? options = null)
    {
        _backendSelector = backendSelector;
        _options = options;
        _placement = options?.Placement ?? PlacementConfig.Single;
        PlacementPlanner.ValidatePlacement(_placement);
        _images = new Lazy<ImagesService>(() => new ImagesService(this));
        _video = new Lazy<VideoService>(() => new VideoService(this));
        _text = new Lazy<TextService>(() => new TextService(this));
        _music = new Lazy<MusicService>(() => new MusicService(this));
        _speech = new Lazy<SpeechService>(() => new SpeechService(this));
        _transcribe = new Lazy<TranscribeService>(() => new TranscribeService(this));
        _voiceConversion = new Lazy<VoiceConversionService>(() => new VoiceConversionService(this));
        _fx = new Lazy<FxService>(() => new FxService(this));
        _vision = new Lazy<VisionService>(() => new VisionService(this));
        _restore = new Lazy<RestoreService>(() => new RestoreService(this));
        _mesh = new Lazy<MeshService>(() => new MeshService(this));
        _world = new Lazy<WorldService>(() => new WorldService(this));
        _embeddings = new Lazy<EmbeddingService>(() => new EmbeddingService(this));
    }

    /// <summary>Creates an engine on a specific device, for hosts that carry the backend name and the GPU id as separate settings.</summary>
    /// <remarks>Equivalent to passing <c>"cuda:1"</c> to the single-argument constructor; ordinal 0 composes back to the plain selector, so
    /// the default path is byte-identical to what callers passed before device selection existed.</remarks>
    public InferenceEngine(string backendSelector, int deviceOrdinal, EngineOptions? options = null)
        : this(BackendFactory.WithOrdinal(backendSelector, deviceOrdinal), options)
    {
    }

    /// <inheritdoc/>
    public string BackendSelector => _backendSelector;

    /// <summary>The device ordinal this engine's backend will be (or was) constructed on; 0 unless the selector carries a suffix.</summary>
    public int DeviceOrdinal => BackendFactory.ParseOrdinal(_backendSelector);

    /// <inheritdoc/>
    public string BackendDescription => BackendFactory.Describe(_backendSelector);

    /// <inheritdoc/>
    public bool IsSupported(Modality modality) => Modalities.All.Contains(modality);

    /// <inheritdoc/>
    public IReadOnlyCollection<string> LoadedPipelineKeys =>
        [.. _recipePipelines.Keys, .. _videoRecipePipelines.Keys];

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

    /// <summary>Video/image restoration (SeedVR2).</summary>
    public IRestoreService Restore => _restore.Value;

    /// <inheritdoc/>
    public IMeshService Mesh => _mesh.Value;

    /// <inheritdoc/>
    public IWorldService World => _world.Value;

    /// <inheritdoc/>
    public IEmbeddingService Embeddings => _embeddings.Value;

    /// <inheritdoc/>
    public void SetBackend(string selector)
    {
        _backendSelector = selector;
        DisposeLoaded();
    }

    /// <summary>This engine's multi-device placement. <see cref="PlacementConfig.Single"/> unless configured.</summary>
    public PlacementConfig Placement => _placement;

    /// <summary>Replaces the placement and drops loaded pipelines (their weights live on the old devices).
    /// The all-defaults config restores exact single-device behavior.</summary>
    public void SetPlacement(PlacementConfig placement)
    {
        ArgumentNullException.ThrowIfNull(placement);
        PlacementPlanner.ValidatePlacement(placement);
        if (_placement == placement)
        {
            return;
        }
        _placement = placement;
        DisposeLoaded();
    }

    /// <summary>The compute backend, constructed on first use. Used by the typed services that drive pipelines directly
    /// (the recipe registry) rather than through a modality handler.</summary>
    internal IBackend Backend => EnsureBackend();

    /// <summary>The engine's compute backend for hosts that run auxiliary work (e.g. ControlNet annotators) on the
    /// same device. The engine owns it — callers must never dispose it; disposal would evict the engine's resident
    /// weights and unbind its compute stream.</summary>
    public IBackend ComputeBackend => EnsureBackend();

    /// <summary>This engine's audio runtime (generation lock + resident runner caches), created on first audio use.
    /// Per-engine so audio on two engines/GPUs runs concurrently and one engine's FreeMemory cannot unload another's
    /// models.</summary>
    internal Audio.AudioRuntime AudioRuntime => _audioRuntime ??= new Audio.AudioRuntime();

    /// <summary>Detects the checkpoint architecture for <paramref name="spec"/>, resolves its recipe, and constructs
    /// (or returns a cached) pipeline. Throws when no recipe is registered for the detected family yet.</summary>
    internal IRecipePipeline GetOrConstructRecipe(ModelSpec spec, ImageRequest? request = null, string? alsoKeepPath = null)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                "No checkpoint found for this model. Pass a checkpoint via --model-path or let the catalog fetch it first.");
        }

        // LoRA and component overrides are baked into the loaded weights, so they are part of the cache identity.
        string key = $"recipe:{spec.LocalPath}|{RecipeCacheKey.Describe(request)}{_placement.CacheKey()}";
        if (_recipePipelines.TryGetValue(key, out IRecipePipeline? cached))
            return cached;

        // Switch-pressure eviction: pipelines for OTHER checkpoints keep their multi-GB weights resident
        // (HARTSY_KEEP_MODELS) and cannot share the card with the incoming model at fleet sizes — a
        // Krea2(13 GB)→Z-Image switch measured 74 MB free on 24 GB before this existed
        // (benchmarks/results/2026-07-23_swarm_stepcache_verification.md §engine bugs).
        EvictOtherCheckpointPipelines(spec.LocalPath, alsoKeepPath);

        IArchitectureRecipe recipe = ResolveRecipe(spec);
        IBackend backend = EnsureBackend();
        IRecipePipeline pipeline = ConstructWithVramCleanup(backend, spec, () => recipe.Construct(new RecipeContext
        {
            CheckpointPath = spec.LocalPath,
            Backend = backend,
            TextEncoderBackend = _placement.TextEncoderDevice is null ? null : EnsureBackend(_placement.TextEncoderDevice),
            VaeBackend = _placement.VaeDevice is null ? null : EnsureBackend(_placement.VaeDevice),
            CfgParallelBackend = EnsureCfgParallelBackend(),
            DitShardBackend = EnsureDitShardBackend(),
            DitShardBackends = EnsureDitShardBackends(),
            CpBackends = EnsureCpBackends(),
            Components = request?.Components,
            Loras = request?.Loras,
        }));
        _recipePipelines[key] = pipeline;
        return pipeline;
    }

    /// <summary>Runs a generation and, when it fails for lack of VRAM, releases every device buffer the engine holds
    /// before rethrowing — so the failure costs one request, not the process.</summary>
    /// <remarks>A pipeline uploads its components phase by phase (text encoder, then DiT, then VAE) and only frees them
    /// on the success path. An <see cref="OutOfVramException"/> partway through therefore leaves every earlier phase
    /// resident with nothing running — the measured symptom was ~11.5 GB held after a failed request, which then made a
    /// *separate* ComfyUI process on the same card fail too. Scoped deliberately to capacity failures: a cancellation or
    /// a validation error must not evict a healthy resident model (HARTSY_KEEP_MODELS) and pay a re-upload for nothing.
    /// The reclaim is best-effort — an exception inside it must never replace the real one the caller needs to see.</remarks>
    internal T GenerateWithVramCleanup<T>(Func<T> generate)
    {
        // Per-device serialization: with two backends sharing one GPU (two SwarmUI backends, or image + LLM
        // slots), state is isolated per backend but concurrent same-device execution is not yet in contract.
        // ALL placement devices are gated (CFG-parallel/TE/VAE/shard stages), not just the primary — a sibling
        // engine gated on one of THOSE ordinals must not run concurrently with the branch/stage work there.
        using IDisposable gate = DeviceGate.AcquireAll(GateBackends());
        try
        {
            return generate();
        }
        catch (OutOfVramException ex)
        {
            Logs.Error("[Engine] Generation ran out of VRAM — releasing device memory held by the partially-loaded pipeline.", ex);
            try
            {
                // Unlike EvictOtherCheckpointPipelines (which falls back to a pool trim while any recipe pipeline is
                // still cached, to avoid nuking a healthy resident model), the full sweep is the right call here: the
                // failing pipeline IS cached, so the guarded branch would never fire, and leaving a half-loaded
                // pipeline's phases resident is the whole defect. Eviction is recoverable — weights re-upload from
                // their host tensors on next use, the same cycle QwenImagePipeline.EvictResidentTransformer relies on.
                _backend?.FreeAllDeviceMemory();
            }
            catch (Exception cleanupEx)
            {
                Logs.Error("[Engine] Post-OOM device cleanup itself failed — the process may still be holding VRAM.", cleanupEx);
            }
            throw;
        }
    }

    /// <summary>Runs a recipe's <c>Construct</c> and, on <b>any</b> failure, releases every device buffer it allocated
    /// before rethrowing.</summary>
    /// <remarks>Recipes upload multi-GB weight sets across several components (text encoder, DiT, VAE) before returning
    /// a pipeline the caller can dispose. When construction throws partway, the half-built pipeline is never stored, so
    /// its weights become permanently unreachable — the <see cref="Tensor"/> keys they are cached under have no
    /// surviving owner, so no <c>FreeWeights</c> call can ever name them again. Measured impact: after one oversized
    /// request the process held ~11.5 GB with nothing running, which starved a *separate* ComfyUI process on the same
    /// card until SwarmUI was killed (benchmarks/results/2026-07-26_image_comfy-vs-hartsy.md, Bug 2).
    /// <para>Deliberately broader than <see cref="GenerateWithVramCleanup"/>, which narrows to
    /// <see cref="OutOfVramException"/>: the reachability argument does not depend on <i>why</i> construction failed. A
    /// missing side-model download, a tokenizer error or a bad checkpoint key thrown after the DiT upload leaks exactly
    /// as much as an OOM does, so every escape from <c>Construct</c> must reclaim. The generate path can afford to be
    /// narrow precisely because its pipeline stays cached and reachable, making a non-capacity failure cost nothing.
    /// Only 16 of 47 image recipes have any exception handling of their own, so this belongs at the single shared
    /// boundary rather than per recipe.</para></remarks>
    private T ConstructWithVramCleanup<T>(IBackend backend, ModelSpec spec, Func<T> construct)
    {
        // Construction uploads multi-GB weight sets — serialize it against a same-device sibling's generation
        // exactly like the generate path (callers run construct and generate sequentially, never nested, so the
        // non-reentrant gate cannot self-deadlock). Gate every placement device: construction resolves TE/VAE/
        // CFG/shard backends inside the construct() lambda and uploads to them too.
        using IDisposable gate = DeviceGate.AcquireAll(GateBackends());
        try
        {
            return construct();
        }
        catch (Exception ex)
        {
            Logs.Error($"[Engine] Recipe construction failed for '{spec.LocalPath}' — releasing device memory it had already allocated.", ex);
            try
            {
                backend.FreeAllDeviceMemory();
            }
            catch (Exception cleanupEx)
            {
                Logs.Error("[Engine] Post-failure device cleanup itself failed — the process may still be holding VRAM.", cleanupEx);
            }
            throw;
        }
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
        => VideoRecipeRegistry.Resolve(ResolveVideoFamilyId(spec))?.Defaults ?? VideoDefaults.Standard;

    /// <summary>Video-path family id: <see cref="ResolveFamilyId"/> plus checkpoint-aware remaps (currently only
    /// LTX-2.5 distilled-by-filename). Video-only — image recipes carry no per-checkpoint contracts by name.</summary>
    internal static string ResolveVideoFamilyId(ModelSpec spec)
        => Recipes.Video.LtxVideo2DistilledRouting.RemapFamilyId(ResolveFamilyId(spec), spec.LocalPath);

    /// <summary>The conditioning the video recipe for <paramref name="spec"/> declares it can apply. Resolved through
    /// the same registry lookup the construction path uses, so it cannot disagree with the pipeline that will run.
    /// Wan is checkpoint-aware: its conditioning variants share the family's compat classes and are detected by
    /// header sniff, exactly like the construction-time delegation.</summary>
    internal static VideoFeatures SupportedVideoFeatures(ModelSpec spec)
    {
        IVideoRecipe? recipe = VideoRecipeRegistry.Resolve(ResolveVideoFamilyId(spec));
        return recipe switch
        {
            null => VideoFeatures.None,
            Recipes.Video.WanVideoRecipe wan => wan.SupportsFor(spec.LocalPath),
            Recipes.Video.LtxVideoRecipe ltx => ltx.SupportsFor(spec.LocalPath),
            _ => recipe.Supports,
        };
    }

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
    internal IVideoRecipePipeline GetOrConstructVideoRecipe(ModelSpec spec, VideoRequest? request = null)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                "No checkpoint found for this model. Pass a checkpoint via --model-path or let the catalog fetch it first.");
        }

        // LoRA and component overrides are baked into the loaded weights, so they are part of the cache identity —
        // the same rule the image path already follows. Without this a LoRA request reuses the un-merged pipeline.
        string key = $"video-recipe:{spec.LocalPath}|{RecipeCacheKey.Describe(request)}{_placement.CacheKey()}";
        if (_videoRecipePipelines.TryGetValue(key, out IVideoRecipePipeline? cached))
            return cached;

        // Same switch-pressure eviction as the image path — video DiTs are the largest residents of all.
        EvictOtherCheckpointPipelines(spec.LocalPath);

        string familyId = ResolveVideoFamilyId(spec);
        IVideoRecipe recipe = VideoRecipeRegistry.Resolve(familyId)
            ?? throw new NotSupportedException(
                $"Video family '{familyId}' has no recipe lifted into the Engine yet (E-IMG-3). " +
                $"Currently drivable: {string.Join(", ", VideoRecipeRegistry.RegisteredNames)}.");

        IBackend backend = EnsureBackend();
        IVideoRecipePipeline pipeline = ConstructWithVramCleanup(backend, spec,
            () => recipe.Construct(new RecipeContext
            {
                CheckpointPath = spec.LocalPath,
                Backend = backend,
                TextEncoderBackend = _placement.TextEncoderDevice is null ? null : EnsureBackend(_placement.TextEncoderDevice),
                VaeBackend = _placement.VaeDevice is null ? null : EnsureBackend(_placement.VaeDevice),
                CfgParallelBackend = EnsureCfgParallelBackend(),
                DitShardBackend = EnsureDitShardBackend(),
                DitShardBackends = EnsureDitShardBackends(),
                CpBackends = EnsureCpBackends(),
                Components = request?.Components,
                Loras = request?.Loras,
                VideoSwapModelPath = string.IsNullOrWhiteSpace(request?.VideoSwapModel) ? null : request!.VideoSwapModel,
                VideoSwapPercent = request?.VideoSwapPercent,
            }));
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

    private IBackend EnsureBackend()
    {
        if (_backend is null)
        {
            _backend = BackendFactory.Create(_backendSelector);
            // Weak-keyed override: it dies with the backend, so rebuilds re-apply here and disposal needs no cleanup.
            if (_options?.LowVram is LowVramMode lowVram)
            {
                LowVramPolicy.SetOverride(_backend, lowVram);
            }
        }
        return _backend;
    }

    /// <summary>The backend for a placement device selector: the primary when the selector resolves to the
    /// engine's own device, else a pooled secondary constructed on first use (with the engine's LowVram override
    /// applied). Null/blank selector = primary.</summary>
    internal IBackend EnsureBackend(string? selector)
    {
        if (string.IsNullOrWhiteSpace(selector))
        {
            return EnsureBackend();
        }
        string canonical = CanonicalSelector(selector);
        if (canonical.Equals(CanonicalSelector(_backendSelector), StringComparison.OrdinalIgnoreCase))
        {
            return EnsureBackend();
        }
        if (_placementBackends.TryGetValue(canonical, out IBackend? existing))
        {
            return existing;
        }
        IBackend created = BackendFactory.Create(canonical);
        if (_options?.LowVram is LowVramMode lowVram)
        {
            LowVramPolicy.SetOverride(created, lowVram);
        }
        _placementBackends[canonical] = created;
        return created;
    }

    /// <summary>Every backend a generation under the current placement can touch — the primary plus each
    /// configured placement device, resolved (and therefore constructed) eagerly so the gate covers them even
    /// before the pipeline's first use creates them. Pooled by canonical selector, so repeats dedupe to the
    /// same instance and <see cref="DeviceGate.AcquireAll"/> dedupes the ordinals.</summary>
    private IEnumerable<IBackend?> GateBackends()
    {
        yield return EnsureBackend();
        if (_placement.TextEncoderDevice is not null)
        {
            yield return EnsureBackend(_placement.TextEncoderDevice);
        }
        if (_placement.VaeDevice is not null)
        {
            yield return EnsureBackend(_placement.VaeDevice);
        }
        if (_placement.CfgParallelDevice is not null)
        {
            yield return EnsureBackend(_placement.CfgParallelDevice);
        }
        if (_placement.EnableDitSharding)
        {
            foreach (string device in _placement.ShardDevices)
            {
                yield return EnsureBackend(device);
            }
        }
        if (_placement.ContextParallelDevices.Count >= 2)
        {
            foreach (string device in _placement.ContextParallelDevices)
            {
                yield return EnsureBackend(device);
            }
        }
    }

    /// <summary>The CFG-branch-parallel backend, or null when unset OR when the selector resolves to the primary
    /// instance. The null-out is load-bearing: <c>CfgBranchRunner</c> drives the cond and uncond branches from two
    /// threads, and a same-device selector would hand both threads ONE backend whose per-<c>State</c> caches are
    /// plain unlocked dictionaries — pipelines gate only on <c>CfgParallelBackend is not null</c>, so this is the
    /// single choke point that keeps that config from ever reaching them.</summary>
    private IBackend? EnsureCfgParallelBackend()
    {
        if (_placement.CfgParallelDevice is null)
        {
            return null;
        }
        IBackend resolved = EnsureBackend(_placement.CfgParallelDevice);
        if (ReferenceEquals(resolved, EnsureBackend()))
        {
            Logs.Warning($"[Engine] CfgParallelDevice '{_placement.CfgParallelDevice}' resolves to the primary "
                + "backend — CFG-branch parallelism disabled (two branch threads must not share one backend).");
            return null;
        }
        return resolved;
    }

    /// <summary>The second backend for Phase 8 DiT sharding (<see cref="PlacementConfig.EnableDitSharding"/>), or
    /// null when it's off. <c>ShardDevices[1]</c> is safe to index unchecked — <see cref="PlacementPlanner.ValidatePlacement"/>
    /// rejects any <see cref="PlacementConfig.EnableDitSharding"/> config with fewer than 2 <c>ShardDevices</c>
    /// at both construction and <see cref="SetPlacement"/>, so an engine can never reach this with fewer. Kept as
    /// the 2-way shape the five non-Qwen sharded recipes still consume; <see cref="EnsureDitShardBackends"/> is the
    /// N-way sibling.</summary>
    private IBackend? EnsureDitShardBackend() =>
        _placement.EnableDitSharding ? EnsureBackend(_placement.ShardDevices[1]) : null;

    /// <summary>Every stage-1..N-1 backend for N-way DiT sharding (stage 0 is always <see cref="EnsureBackend()"/>),
    /// or null when sharding is off. Note <c>EnsureBackend(selector)</c> pools by canonical device identity, so a
    /// <c>ShardDevices</c> entry that canonically matches an already-resolved stage (e.g. two entries that both
    /// resolve to "cuda:0") returns the SAME backend instance for both — two logically distinct stages then share
    /// one physical GPU AND one backend object, which <c>QwenImageTransformer.ForwardSharded</c> must (and does)
    /// handle as a same-instance no-op copy at that boundary. Getting two INDEPENDENT backend instances on one
    /// physical GPU (the same-GPU-multi-backend feature) requires two separate engines, not two entries in one
    /// engine's <c>ShardDevices</c> — see <c>SameGpuConcurrentRealWeightTests</c>.</summary>
    private IReadOnlyList<IBackend>? EnsureDitShardBackends()
    {
        if (!_placement.EnableDitSharding || _placement.ShardDevices.Count < 2)
        {
            return null;
        }
        List<IBackend> stages = new(_placement.ShardDevices.Count - 1);
        for (int i = 1; i < _placement.ShardDevices.Count; i++)
        {
            stages.Add(EnsureBackend(_placement.ShardDevices[i]));
        }
        return stages;
    }

    /// <summary>The ordered context-parallel rank backends, or null when unset OR when the list is unusable: rank 0
    /// must resolve to the primary instance (the pipeline runs rank 0 on <see cref="RecipeContext.Backend"/>), and no
    /// two entries may resolve to ONE backend instance — two rank threads driving one backend's plain unlocked
    /// per-<c>State</c> caches is the same hazard <see cref="EnsureCfgParallelBackend"/> nulls out. Pipelines gate
    /// only on <c>CpBackends is not null</c>, so this is the single choke point.</summary>
    private IReadOnlyList<IBackend>? EnsureCpBackends()
    {
        if (_placement.ContextParallelDevices.Count < 2)
        {
            return null;
        }
        List<IBackend> ranks = new(_placement.ContextParallelDevices.Count);
        foreach (string device in _placement.ContextParallelDevices)
        {
            ranks.Add(EnsureBackend(device));
        }
        if (!ReferenceEquals(ranks[0], EnsureBackend()))
        {
            Logs.Warning($"[Engine] ContextParallelDevices[0] '{_placement.ContextParallelDevices[0]}' does not "
                + "resolve to the primary backend — context parallelism disabled (rank 0 must be the primary).");
            return null;
        }
        for (int i = 0; i < ranks.Count; i++)
        {
            for (int j = i + 1; j < ranks.Count; j++)
            {
                if (ReferenceEquals(ranks[i], ranks[j]))
                {
                    Logs.Warning($"[Engine] ContextParallelDevices entries {i} and {j} resolve to the same backend "
                        + "instance — context parallelism disabled (two rank threads must not share one backend).");
                    return null;
                }
            }
        }
        return ranks;
    }

    /// <summary>Canonical device identity for pooling: resolved kind plus ordinal ("cuda:1", "cuda", "cpu"), so
    /// "auto", "cuda:0" and "cuda" on a CUDA box all pool to the same backend.</summary>
    private static string CanonicalSelector(string selector) =>
        BackendFactory.WithOrdinal(BackendFactory.Resolve(selector), BackendFactory.ParseOrdinal(selector));

    /// <summary>Disposes every cached image/video pipeline bound to a checkpoint OTHER than
    /// <paramref name="keepPath"/> and returns their device memory to the driver. Called on a cache miss
    /// before constructing the incoming model, so a model SWITCH can never stack resident weight sets until
    /// the card OOMs (and poisons the session with cuDNN/VAE fallbacks). Pipelines for the SAME checkpoint
    /// (LoRA/component variants) are kept. Deliberately does NOT touch the other services (Text/Audio/…)
    /// — their memory is managed by their own slots, and <see cref="FreeMemory"/> remains the full sweep.</summary>
    /// <param name="keepPath">The incoming checkpoint — never evicted.</param>
    /// <param name="alsoKeepPath">Optional second survivor. The generic refiner sets this in both directions
    /// (base keeps refiner, refiner keeps base): without it every refined generation ping-pongs two full
    /// pipeline loads — the base's construct evicts the refiner, the refine evicts the base — which besides the
    /// load time churns a fresh host-side weight copy per swap.</param>
    private void EvictOtherCheckpointPipelines(string keepPath, string? alsoKeepPath = null)
    {
        string keepImagePrefix = $"recipe:{keepPath}|";
        string keepVideoPrefix = $"video-recipe:{keepPath}|";
        string? keepImagePrefix2 = alsoKeepPath is null ? null : $"recipe:{alsoKeepPath}|";
        string? keepVideoPrefix2 = alsoKeepPath is null ? null : $"video-recipe:{alsoKeepPath}|";
        List<string> imageVictims = _recipePipelines.Keys
            .Where(k => !k.StartsWith(keepImagePrefix, StringComparison.OrdinalIgnoreCase)
                && (keepImagePrefix2 is null || !k.StartsWith(keepImagePrefix2, StringComparison.OrdinalIgnoreCase))).ToList();
        List<string> videoVictims = _videoRecipePipelines.Keys
            .Where(k => !k.StartsWith(keepVideoPrefix, StringComparison.OrdinalIgnoreCase)
                && (keepVideoPrefix2 is null || !k.StartsWith(keepVideoPrefix2, StringComparison.OrdinalIgnoreCase))).ToList();
        if (imageVictims.Count == 0 && videoVictims.Count == 0)
        {
            return;
        }

        foreach (string victim in imageVictims)
        {
            _recipePipelines[victim].Dispose();
            _recipePipelines.Remove(victim);
        }
        foreach (string victim in videoVictims)
        {
            _videoRecipePipelines[victim].Dispose();
            _videoRecipePipelines.Remove(victim);
        }
        // Disposal only drops the PIPELINE's references — the backend's device weight cache still holds the
        // promoted copies (keyed by Tensor, strong refs), so Dispose+GC+Trim alone freed ~nothing (measured
        // 2026-07-23: Z-Image still at 0.4% free after "evicting" Krea2's 13 GB). When the switch leaves NO
        // cached recipe pipelines (the normal one-model-at-a-time case), sweep the backend's caches outright —
        // weights/casts/activations all belonged to the victims. When same-checkpoint variants survive, the
        // sweep would nuke their resident weights too, so fall back to the soft release (they re-upload).
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        try
        {
            if (_recipePipelines.Count == 0 && _videoRecipePipelines.Count == 0)
            {
                _backend?.FreeAllDeviceMemory();
            }
            else
            {
                _backend?.TrimMemoryPool();
            }
        }
        catch (Exception ex)
        {
            Logs.Warning($"[Engine] Device release after model-switch eviction failed: {ex.Message}");
        }
        // The victims' weights also live on any placement backends they used (CFG-parallel replica, TE/VAE,
        // shard stages) — without this sweep each model switch stacked another full replica on the second GPU.
        foreach (IBackend extra in _placementBackends.Values)
        {
            try
            {
                if (_recipePipelines.Count == 0 && _videoRecipePipelines.Count == 0)
                {
                    extra.FreeAllDeviceMemory();
                }
                else
                {
                    extra.TrimMemoryPool();
                }
            }
            catch (Exception ex)
            {
                Logs.Warning($"[Engine] Placement-backend release after model-switch eviction failed: {ex.Message}");
            }
        }
        // Host side: the victims' weight buffers are freed by now, but glibc keeps their arena heaps resident,
        // so family rotation ratchets anon RSS into the cgroup cap. See HostMemory.Trim.
        HostMemory.TrimAndLog("model-switch eviction");
        Logs.Info($"[Engine] Model switch: evicted {imageVictims.Count + videoVictims.Count} resident pipeline(s) for other checkpoints.");
    }

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
        // RestoreService caches the SeedVR2 pipeline + mmap-backed weight loaders bound to the backend.
        if (_restore.IsValueCreated)
        {
            _restore.Value.ReleasePipeline();
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
        // EmbeddingService's cached DecoderEmbeddingModels hold device-resident weights bound to the backend
        // being torn down/switched — same reasoning as TextService above.
        if (_embeddings.IsValueCreated)
        {
            _embeddings.Value.Dispose();
        }
        // Audio pipelines are cached per-engine by the audio runtime; drop THIS engine's so none outlives the backend
        // it was constructed against. Other engines' resident audio models are untouched.
        _audioRuntime?.UnloadAll(AudioUnloadWaitSeconds);
        if (disposeBackend)
        {
            foreach (IBackend extra in _placementBackends.Values)
            {
                extra.Dispose();
            }
            _placementBackends.Clear();
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
        foreach (IBackend extra in _placementBackends.Values)
        {
            try
            {
                extra.FreeAllDeviceMemory();
                extra.TrimMemoryPool();
            }
            catch (Exception ex)
            {
                Logs.Warning($"[Engine] Releasing a placement backend's device memory failed: {ex.Message}");
            }
        }
        HostMemory.TrimAndLog("free-memory sweep");
    }

    /// <inheritdoc/>
    public void Dispose() => DisposeLoaded();
}
