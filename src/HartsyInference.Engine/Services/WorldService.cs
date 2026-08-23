using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.Diamond;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.World.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Services;

/// <summary>Interactive-world service: routes to a per-model pipeline by catalog id, caches it per checkpoint, and hands out sessions that drive it. <c>oasis</c> and <c>diamond</c> are fully wired. <c>hunyuan-gamecraft</c> has a real, code-complete checkpoint loader (<see cref="LoadHunyuanGameCraft"/> — see its remarks) but still fails fast with a clear, specific message today: it needs a VAE encoder no converter builds yet, so no session could ever succeed regardless of weights present. <c>matrix-game-2</c> and <c>matrix-game-3</c> are catalogued but not yet loadable here — each needs a multi-checkpoint loader (transformer + VAE + CLIP/T5 text encoder) this service does not build yet, and Matrix-Game 3.0 additionally has no image→latent encoder ported (<c>Wan22VaeEncoder</c> is decode-only today). Selecting any of those three fails fast with a clear message instead of silently mis-loading as Oasis.</summary>
public sealed class WorldService : IWorldService, IDisposable
{
    /// <summary>Aux key on <see cref="ModelSpec"/> that points at the model's VAE checkpoint (Oasis's ViT-VAE; Hunyuan-GameCraft's 3D causal VAE).</summary>
    public const string VaeAuxKey = "vae-path";

    /// <summary>Aux key on <see cref="ModelSpec"/> that points at Hunyuan-GameCraft's Llava-Llama-3-8B text-encoder checkpoint.</summary>
    public const string LlavaAuxKey = "llava-path";

    /// <summary>Aux key on <see cref="ModelSpec"/> that points at Hunyuan-GameCraft's CLIP-L text-encoder checkpoint.</summary>
    public const string ClipAuxKey = "clip-path";

    private readonly InferenceEngine _engine;
    private readonly Dictionary<string, LoadedWorld> _pipelines = new(StringComparer.OrdinalIgnoreCase);
    private readonly object _gate = new object();

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal WorldService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public IWorldSession Open(ModelSpec spec, WorldRequest request)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(request);
        if (request.InitImage is null)
        {
            throw new ArgumentException("World models roll out from a first frame; set WorldRequest.InitImage.", nameof(request));
        }
        LoadedWorld loaded = GetOrLoad(spec);
        return loaded.OpenSession(request);
    }

    /// <summary>Loads (or returns the cached) world pipeline for <paramref name="spec"/>, chosen by catalog id.</summary>
    private LoadedWorld GetOrLoad(ModelSpec spec)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException("No world-model checkpoint found. Pass it as the spec's local path.");
        }
        string id = (spec.Catalog?.Id ?? "oasis").ToLowerInvariant();
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
        if (!spec.Aux.TryGetValue(VaeAuxKey, out string? vaePath))
        {
            throw new ArgumentException($"Oasis needs its ViT-VAE checkpoint via the '{VaeAuxKey}' aux path.", nameof(spec));
        }
        try
        {
            (Dictionary<string, Core.Tensors.Tensor> ditWeights, SafeTensorsLoader ditLoader) = OasisCheckpointConverter.LoadAndConvert(spec.LocalPath!);
            OasisDit dit = new OasisDit(new OasisDitConfig());
            dit.LoadWeights(ditWeights);

            (Dictionary<string, Core.Tensors.Tensor> vaeWeights, SafeTensorsLoader vaeLoader) = OasisCheckpointConverter.LoadAndConvert(vaePath);
            OasisVitVae vae = new OasisVitVae();
            vae.LoadWeights(vaeWeights);

            // VaeDevice overlaps a finished frame's VAE decode with the next frame's denoise on the primary —
            // see OasisPipeline.Generate's overlapDecode path. Unset defaults VaeBackend to Backend (today's behavior).
            OasisPipeline pipeline = new OasisPipeline(_engine.Backend, dit, vae)
            {
                VaeBackend = _engine.EnsureBackend(_engine.Placement.VaeDevice),
            };
            return new LoadedWorld(pipeline, [ditLoader, vaeLoader], request => new OasisWorldSession(pipeline, request));
        }
        catch (Exception ex)
        {
            Logs.Error($"Failed to load the Oasis world model from {spec.LocalPath}", ex);
            throw;
        }
    }

    /// <summary>Loads DIAMOND's denoiser from the spec local path — a single checkpoint file, no aux paths. Genuinely per-frame interactive; see <see cref="DiamondWorldSession"/>.</summary>
    private LoadedWorld LoadDiamond(ModelSpec spec)
    {
        try
        {
            DiamondWorldPipeline pipeline = DiamondWorldPipeline.LoadFromPath(_engine.Backend, spec.LocalPath!, DiamondConfig.Atari(4));
            return new LoadedWorld(pipeline, [], request => new DiamondWorldSession(pipeline, request));
        }
        catch (Exception ex)
        {
            Logs.Error($"Failed to load the DIAMOND world model from {spec.LocalPath}", ex);
            throw;
        }
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
        if (!spec.Aux.TryGetValue(VaeAuxKey, out string? vaePath))
        {
            throw new ArgumentException($"Hunyuan-GameCraft needs its 3D-VAE checkpoint via the '{VaeAuxKey}' aux path.", nameof(spec));
        }
        if (!spec.Aux.TryGetValue(LlavaAuxKey, out string? llavaPath))
        {
            throw new ArgumentException($"Hunyuan-GameCraft needs its Llava-Llama-3-8B checkpoint via the '{LlavaAuxKey}' aux path.", nameof(spec));
        }
        if (!spec.Aux.TryGetValue(ClipAuxKey, out string? clipPath))
        {
            throw new ArgumentException($"Hunyuan-GameCraft needs its CLIP-L checkpoint via the '{ClipAuxKey}' aux path.", nameof(spec));
        }
        if (!HunyuanGameCraftPipeline.LoadFromPathBuildsVaeEncoder)
        {
            // No session opened against a LoadFromPath pipeline can ever succeed today (see
            // HunyuanGameCraftPipeline.CanEncodeReferenceFrame's remarks) — fail here, BEFORE loading the ~51GB
            // checkpoint set (DiT+CameraNet ~30GB, Llava-Llama-3-8B ~16GB, CLIP-L, VAE) into RAM for nothing. This
            // box has hit real 45-62GB OOM kills before; don't add a new way to get there.
            throw new NotSupportedException(HunyuanGameCraftWorldSession.MissingVaeEncoderMessage);
        }

        HunyuanGameCraftPipeline? pipeline = null;
        List<IDisposable> owned = [];
        try
        {
            pipeline = HunyuanGameCraftPipeline.LoadFromPath(_engine.Backend, spec.LocalPath!, vaePath);

            (Dictionary<string, Core.Tensors.Tensor> llavaWeights, IDisposable llavaLoader) = HunyuanGameCraftCheckpointConverter.LoadRaw(llavaPath);
            owned.Add(llavaLoader);
            LlamaStyleEncoder llava = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Llama31_8B);
            llava.LoadWeights(llavaWeights);
            owned.Add(llava);

            (Dictionary<string, Core.Tensors.Tensor> clipWeights, IDisposable clipLoader) = HunyuanGameCraftCheckpointConverter.LoadRaw(clipPath);
            owned.Add(clipLoader);
            ClipTextEncoder clipL = new ClipTextEncoder(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipWeights, "text_model");

            ClipTokenizer clipTokenizer = new ClipTokenizer();
            owned.Add(clipTokenizer);

            return new LoadedWorld(pipeline, owned, request => new HunyuanGameCraftWorldSession(pipeline, llava, clipL, clipTokenizer, request));
        }
        catch (Exception ex)
        {
            Logs.Error($"Failed to load the Hunyuan-GameCraft world model from {spec.LocalPath}", ex);
            pipeline?.Dispose();
            foreach (IDisposable d in owned)
            {
                d.Dispose();
            }
            throw;
        }
    }

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

    /// <summary>A loaded world pipeline plus its session factory and the loaders that own its mmap-backed weights.</summary>
    private sealed class LoadedWorld : IDisposable
    {
        private readonly IDisposable _pipeline;
        private readonly IReadOnlyList<IDisposable> _owned;
        private readonly Func<WorldRequest, IWorldSession> _openSession;

        /// <summary>Wraps <paramref name="pipeline"/>, the loaders whose lifetime it depends on, and its session factory.</summary>
        public LoadedWorld(IDisposable pipeline, IReadOnlyList<IDisposable> owned, Func<WorldRequest, IWorldSession> openSession)
        {
            _pipeline = pipeline;
            _owned = owned;
            _openSession = openSession;
        }

        /// <summary>Opens a new session over the loaded pipeline.</summary>
        public IWorldSession OpenSession(WorldRequest request) => _openSession(request);

        /// <inheritdoc/>
        public void Dispose()
        {
            _pipeline.Dispose();
            foreach (IDisposable owned in _owned)
            {
                owned.Dispose();
            }
        }
    }
}
