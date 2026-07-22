using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.Diamond;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.World.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Services;

/// <summary>Interactive-world service: routes to a per-model pipeline by catalog id, caches it per checkpoint, and
/// hands out sessions that drive it. <c>oasis</c> and <c>diamond</c> are fully wired; <c>matrix-game-2</c>,
/// <c>matrix-game-3</c>, and <c>hunyuan-gamecraft</c> are catalogued but not yet loadable here — each needs a
/// multi-checkpoint loader (transformer + VAE + CLIP/T5 text encoder) this service does not build yet, and
/// Matrix-Game 3.0 additionally has no image→latent encoder ported (<c>Wan22VaeEncoder</c> is decode-only today).
/// Selecting one of those three fails fast with a clear message instead of silently mis-loading as Oasis.</summary>
public sealed class WorldService : IWorldService, IDisposable
{
    /// <summary>Aux key on <see cref="ModelSpec"/> that points at the Oasis ViT-VAE checkpoint.</summary>
    public const string VaeAuxKey = "vae-path";

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
            case "hunyuan-gamecraft":
                throw new NotSupportedException(
                    "hunyuan-gamecraft is catalogued but not yet loadable: its checkpoint set is ~51GB minimum (DiT + "
                    + "Llava-Llama3-8B + CLIP-ViT-L + 3D VAE) and this engine has no multi-checkpoint loader for it yet.");
        }
        lock (_gate)
        {
            string key = $"{id}|{spec.LocalPath}|{(spec.Aux.TryGetValue(VaeAuxKey, out string? v) ? v : "")}";
            if (_pipelines.TryGetValue(key, out LoadedWorld? cached))
            {
                return cached;
            }
            LoadedWorld loaded = id == "diamond" ? LoadDiamond(spec) : LoadOasis(spec);
            _pipelines[key] = loaded;
            return loaded;
        }
    }

    /// <summary>Loads the Oasis DiT (spec local path) + its ViT-VAE (aux <see cref="VaeAuxKey"/>). The Oasis
    /// pipeline is a one-shot rollout over a full action plan, not a resumable step API — see
    /// <see cref="OasisWorldSession"/> for exactly what that means for a session.</summary>
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

            OasisPipeline pipeline = new OasisPipeline(_engine.Backend, dit, vae);
            return new LoadedWorld(pipeline, [ditLoader, vaeLoader], request => new OasisWorldSession(pipeline, request));
        }
        catch (Exception ex)
        {
            Logs.Error($"Failed to load the Oasis world model from {spec.LocalPath}", ex);
            throw;
        }
    }

    /// <summary>Loads DIAMOND's denoiser from the spec local path — a single checkpoint file, no aux paths.
    /// Genuinely per-frame interactive; see <see cref="DiamondWorldSession"/>.</summary>
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
