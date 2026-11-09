using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.Diamond;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Worlds;
using HartsyInference.World.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Services;

/// <summary>Interactive-world service: routes to a per-model pipeline by catalog id, caches it per checkpoint, and hands out sessions that drive it. <c>oasis</c> and <c>diamond</c> are fully wired. <c>hunyuan-gamecraft</c> has a real, code-complete checkpoint loader (<see cref="LoadHunyuanGameCraft"/> — see its remarks) but still fails fast with a clear, specific message today: it needs a VAE encoder no converter builds yet, so no session could ever succeed regardless of weights present. <c>matrix-game-2</c> and <c>matrix-game-3</c> are catalogued but not yet loadable here — each needs a multi-checkpoint loader (transformer + VAE + CLIP/T5 text encoder) this service does not build yet, and Matrix-Game 3.0 additionally has no image→latent encoder ported (<c>Wan22VaeEncoder</c> is decode-only today). Selecting any of those three fails fast with a clear message instead of silently mis-loading as Oasis.</summary>
public sealed class WorldService : IWorldService, IDisposable
{
    /// <summary>Aux key on <see cref="ModelSpec"/> that points at the model's VAE checkpoint (Oasis's ViT-VAE; Hunyuan-GameCraft's 3D causal VAE).</summary>
    public const string VaeAuxKey = WorldComponentResolver.VaeAuxKey;

    /// <summary>Aux key on <see cref="ModelSpec"/> that points at Hunyuan-GameCraft's Llava-Llama-3-8B text-encoder checkpoint.</summary>
    public const string LlavaAuxKey = WorldComponentResolver.LlavaAuxKey;

    /// <summary>Aux key on <see cref="ModelSpec"/> that points at Hunyuan-GameCraft's CLIP-L text-encoder checkpoint.</summary>
    public const string ClipAuxKey = WorldComponentResolver.ClipAuxKey;

    private readonly InferenceEngine _engine;
    private readonly Dictionary<string, LoadedWorld> _pipelines = new(StringComparer.OrdinalIgnoreCase);
    private readonly object _gate = new object();
    private readonly Action<ModelSpec>? _onLoad;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal WorldService(InferenceEngine engine)
        : this(engine, null)
    {
    }

    /// <summary>Creates the service with a hook that runs immediately before a checkpoint is loaded; tests use it to count loads.</summary>
    internal WorldService(InferenceEngine engine, Action<ModelSpec>? onLoad)
    {
        _engine = engine;
        _onLoad = onLoad;
    }

    /// <summary>Models held by the load cache. Each keeps its load phase owned until this service is disposed, so the count is the
    /// retained-cache figure, not a leak.</summary>
    internal int RetainedModelCount
    {
        get
        {
            lock (_gate)
            {
                return _pipelines.Count;
            }
        }
    }

    /// <inheritdoc/>
    public IWorldSession Open(ModelSpec spec, WorldRequest request)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(request);
        // Planning refuses an unsupported request before GetOrLoad touches a checkpoint.
        WorldPlan plan = WorldPlanResolver.Resolve(spec, request, _engine.BackendSelector);
        LoadedWorld loaded = GetOrLoad(spec);
        return loaded.OpenSession(plan.Request);
    }

    /// <summary>Loads (or returns the cached) world pipeline for <paramref name="spec"/>, chosen by catalog id.</summary>
    private LoadedWorld GetOrLoad(ModelSpec spec)
    {
        string id = (spec.Catalog?.Id ?? "oasis").ToLowerInvariant();
        // Ordered before the checkpoint guard: these two are not loadable at any path, so the generic
        // "no checkpoint" line would send the caller hunting for weights that cannot help them.
        switch (id)
        {
            case "matrix-game-2":
                throw new NotSupportedException(
                    "matrix-game-2 is catalogued but not yet loadable: it needs a transformer + Wan2.1 VAE encoder/decoder "
                    + "+ OpenCLIP xlm-roberta-ViT-H visual-context loader (~9GB minimum checkpoint set) that this engine "
                    + "does not assemble yet, and its ActionModule numerics are still validation-pending upstream.");
            case "matrix-game-3":
                throw new NotSupportedException(
                    "matrix-game-3 is catalogued but not yet loadable: its checkpoint set is ~27GB minimum, and it has "
                    + "no image-to-latent encoder ported (Wan22VaeEncoder is decode-only today) — the seed image cannot "
                    + "reach the pipeline yet even with weights present.");
        }
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                $"No checkpoint found for world model '{spec.Requested}'. Supply its local path.");
        }
        lock (_gate)
        {
            // Every aux path participates in the cache key — two specs sharing a LocalPath but differing in, say,
            // GameCraft's llava-path/clip-path must not collide on a stale cached pipeline.
            string auxKey = string.Join(';', spec.Aux.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => $"{kv.Key}={kv.Value}"));
            string key = $"{id}|{spec.LocalPath}|{auxKey}";
            if (_pipelines.TryGetValue(key, out LoadedWorld? cached))
            {
                return cached;
            }
            _onLoad?.Invoke(spec);
            LoadedWorld loaded = id switch
            {
                "diamond" => LoadDiamond(spec),
                "hunyuan-gamecraft" => LoadHunyuanGameCraft(spec),
                _ => LoadOasis(spec),
            };
            _pipelines[key] = loaded;
            return loaded;
        }
    }

    /// <summary>Loads the Oasis DiT (spec local path) + its ViT-VAE (aux <see cref="VaeAuxKey"/>). The Oasis pipeline is a one-shot rollout over a full action plan, not a resumable step API — see <see cref="OasisWorldSession"/> for exactly what that means for a session.</summary>
    private LoadedWorld LoadOasis(ModelSpec spec)
    {
        WorldComponents components = WorldComponentResolver.Resolve(spec);
        return LoadInPhase("Oasis", spec, phase =>
        {
            (Dictionary<string, Core.Tensors.Tensor> ditWeights, SafeTensorsLoader ditLoader) =
                OasisCheckpointConverter.LoadAndConvert(components.Backbone);
            phase.Own(ditLoader);
            OasisDit dit = phase.Own(new OasisDit(new OasisDitConfig()));
            dit.LoadWeights(ditWeights);
            phase.OwnRelease("the DiT weight mapping", ditWeights.Clear);

            (Dictionary<string, Core.Tensors.Tensor> vaeWeights, SafeTensorsLoader vaeLoader) =
                OasisCheckpointConverter.LoadAndConvert(components.Vae ?? throw MissingComponent(components));
            phase.Own(vaeLoader);
            OasisVitVae vae = new OasisVitVae();
            vae.LoadWeights(vaeWeights);
            phase.OwnRelease("the VAE weight mapping", vaeWeights.Clear);

            // VaeDevice overlaps a finished frame's VAE decode with the next frame's denoise on the primary —
            // see OasisPipeline.Generate's overlapDecode path. Unset defaults VaeBackend to Backend (today's behavior).
            OasisPipeline pipeline = phase.Own(new OasisPipeline(_engine.Backend, dit, vae)
            {
                VaeBackend = _engine.EnsureBackend(_engine.Placement.VaeDevice),
            });
            return new LoadedWorld(phase, request => new OasisWorldSession(pipeline, request));
        });
    }

    /// <summary>Loads DIAMOND's denoiser from the spec local path — a single checkpoint file, no aux paths. Genuinely per-frame interactive; see <see cref="DiamondWorldSession"/>.</summary>
    private LoadedWorld LoadDiamond(ModelSpec spec)
    {
        WorldComponents components = WorldComponentResolver.Resolve(spec);
        return LoadInPhase("DIAMOND", spec, phase =>
        {
            DiamondWorldPipeline pipeline = phase.Own(
                DiamondWorldPipeline.LoadFromPath(_engine.Backend, components.Backbone, DiamondConfig.Atari(4)));
            return new LoadedWorld(phase, request => new DiamondWorldSession(pipeline, request));
        });
    }

    /// <summary>Would load Hunyuan-GameCraft's DiT+CameraNet+3D-VAE from the spec local path (the merged
    /// <c>mp_rank_00_model_states.pt</c>-style checkpoint plus <see cref="VaeAuxKey"/>), and its Llava-Llama-3-8B
    /// + CLIP-L text encoders from <see cref="LlavaAuxKey"/> / <see cref="ClipAuxKey"/> — mirroring the same
    /// standalone-side-model construction <c>HunyuanVideoRecipe</c> uses for the base HunyuanVideo T2V model — but
    /// currently always throws instead: <see cref="HunyuanGameCraftPipeline.LoadFromPathBuildsVaeEncoder"/> is
    /// false, so no session opened against a loaded pipeline could ever succeed (see
    /// <see cref="HunyuanGameCraftWorldSession"/>'s remarks), and loading the ~51GB checkpoint set for one anyway
    /// would be pure waste (this box has hit real 45-62GB OOM kills before). Aux-key validation still runs first —
    /// it's real, reachable validation independent of the VAE-encoder gap.
    /// <para><b>Loader is code-complete but NOT verified against real weights</b> — this box has no local copy of
    /// the ~51GB checkpoint set (DiT+CameraNet + Llava-Llama3-8B + CLIP-L + 3D VAE) to run it against; see
    /// <c>docs/Checklists/MODEL_STATUS_WORLD.md</c>. <see cref="HunyuanGameCraftPipeline.LoadFromPath"/> and this
    /// method's Llava/CLIP loading are exercised directly by <c>HunyuanGameCraftLoaderTests</c> (bypassing the
    /// gate above) to prove they fail cleanly on missing files rather than crashing.</para></summary>
    private LoadedWorld LoadHunyuanGameCraft(ModelSpec spec)
    {
        WorldComponents components = WorldComponentResolver.Resolve(spec);
        if (!HunyuanGameCraftPipeline.LoadFromPathBuildsVaeEncoder)
        {
            // No session opened against a LoadFromPath pipeline can ever succeed today (see
            // HunyuanGameCraftPipeline.CanEncodeReferenceFrame's remarks) — fail here, BEFORE loading the ~51GB
            // checkpoint set (DiT+CameraNet ~30GB, Llava-Llama-3-8B ~16GB, CLIP-L, VAE) into RAM for nothing. This
            // box has hit real 45-62GB OOM kills before; don't add a new way to get there.
            throw new NotSupportedException(HunyuanGameCraftWorldSession.MissingVaeEncoderMessage);
        }

        return LoadInPhase("Hunyuan-GameCraft", spec, phase =>
        {
            HunyuanGameCraftPipeline pipeline = phase.Own(HunyuanGameCraftPipeline.LoadFromPath(
                _engine.Backend, components.Backbone, components.Vae ?? throw MissingComponent(components)));

            (Dictionary<string, Core.Tensors.Tensor> llavaWeights, IDisposable llavaLoader) =
                HunyuanGameCraftCheckpointConverter.LoadRaw(components.TextEncoder ?? throw MissingComponent(components));
            phase.Own(llavaLoader);
            LlamaStyleEncoder llava = phase.Own(new LlamaStyleEncoder(LlamaStyleEncoderConfig.Llama31_8B));
            llava.LoadWeights(llavaWeights);
            phase.OwnRelease("the Llava weight mapping", llavaWeights.Clear);

            (Dictionary<string, Core.Tensors.Tensor> clipWeights, IDisposable clipLoader) =
                HunyuanGameCraftCheckpointConverter.LoadRaw(components.ClipEncoder ?? throw MissingComponent(components));
            phase.Own(clipLoader);
            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipWeights, "text_model");
            phase.OwnRelease("the CLIP-L weight mapping", clipWeights.Clear);

            ClipTokenizer clipTokenizer = phase.Own(new ClipTokenizer());
            return new LoadedWorld(phase, request => new HunyuanGameCraftWorldSession(pipeline, llava, clipL, clipTokenizer, request));
        });
    }

    /// <summary>Runs one model load under a fresh <see cref="WorldPhase"/>. A load that throws releases everything it acquired
    /// before the exception leaves; a load that returns hands the phase to the cache entry it built.</summary>
    internal static T LoadInPhase<T>(string displayName, ModelSpec spec, Func<WorldPhase, T> load)
    {
        WorldPhase phase = new WorldPhase($"load {displayName}");
        bool committed = false;
        try
        {
            T loaded = load(phase);
            committed = true;
            return loaded;
        }
        catch (Exception ex)
        {
            Logs.Error($"Failed to load the {displayName} world model from {spec.LocalPath}", ex);
            throw;
        }
        finally
        {
            if (!committed)
            {
                phase.Release();
            }
        }
    }

    /// <summary>The resolver names every component an architecture needs, so a null here is a broken resolver, not a bad request.</summary>
    private static InvalidOperationException MissingComponent(WorldComponents components) =>
        new InvalidOperationException($"The {components.Architecture} resolution carries no path for a component it needs.");

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_gate)
        {
            foreach (LoadedWorld loaded in _pipelines.Values)
            {
                loaded.Dispose();
            }
            _pipelines.Clear();
        }
    }

    /// <summary>A loaded world pipeline plus its session factory. The cache entry owns the load phase that built it, so disposing
    /// the entry releases the pipeline and every loader it depends on, pipeline first.</summary>
    private sealed class LoadedWorld : IDisposable
    {
        private readonly WorldPhase _phase;
        private readonly Func<WorldRequest, IWorldSession> _openSession;

        /// <summary>Takes ownership of <paramref name="phase"/>, which holds the pipeline and the loaders it depends on.</summary>
        public LoadedWorld(WorldPhase phase, Func<WorldRequest, IWorldSession> openSession)
        {
            ArgumentNullException.ThrowIfNull(phase);
            ArgumentNullException.ThrowIfNull(openSession);
            _phase = phase;
            _openSession = openSession;
        }

        /// <summary>Opens a new session over the loaded pipeline.</summary>
        public IWorldSession OpenSession(WorldRequest request) => _openSession(request);

        /// <inheritdoc/>
        public void Dispose() => _phase.Release();
    }
}
