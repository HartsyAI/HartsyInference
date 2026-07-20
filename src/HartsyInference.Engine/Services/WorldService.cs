using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.World.Pipelines;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Services;

/// <summary>Interactive-world service over Oasis: loads the DiT (spec local path) and its ViT-VAE (aux
/// <see cref="VaeAuxKey"/>), caches the pipeline per checkpoint, and hands out sessions that drive it. The Oasis
/// pipeline is a one-shot rollout over a full action plan, not a resumable step API — see
/// <see cref="OasisWorldSession"/> for exactly what that means for a session.</summary>
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
            throw new ArgumentException("Oasis rolls out from a first frame; set WorldRequest.InitImage.", nameof(request));
        }
        LoadedWorld loaded = GetOrLoad(spec);
        return new OasisWorldSession(loaded.Pipeline, request);
    }

    /// <summary>Loads (or returns the cached) Oasis pipeline for <paramref name="spec"/>.</summary>
    private LoadedWorld GetOrLoad(ModelSpec spec)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException("No Oasis DiT checkpoint found. Pass it as the spec's local path.");
        }
        if (!spec.Aux.TryGetValue(VaeAuxKey, out string? vaePath))
        {
            throw new ArgumentException($"Oasis needs its ViT-VAE checkpoint via the '{VaeAuxKey}' aux path.", nameof(spec));
        }
        lock (_gate)
        {
            string key = $"{spec.LocalPath}|{vaePath}";
            if (_pipelines.TryGetValue(key, out LoadedWorld? cached))
            {
                return cached;
            }
            try
            {
                (Dictionary<string, Core.Tensors.Tensor> ditWeights, SafeTensorsLoader ditLoader) = OasisCheckpointConverter.LoadAndConvert(spec.LocalPath);
                OasisDit dit = new OasisDit(new OasisDitConfig());
                dit.LoadWeights(ditWeights);

                (Dictionary<string, Core.Tensors.Tensor> vaeWeights, SafeTensorsLoader vaeLoader) = OasisCheckpointConverter.LoadAndConvert(vaePath);
                OasisVitVae vae = new OasisVitVae();
                vae.LoadWeights(vaeWeights);

                LoadedWorld loaded = new LoadedWorld(new OasisPipeline(_engine.Backend, dit, vae), [ditLoader, vaeLoader]);
                _pipelines[key] = loaded;
                return loaded;
            }
            catch (Exception ex)
            {
                Logs.Error($"Failed to load the Oasis world model from {spec.LocalPath}", ex);
                throw;
            }
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

    /// <summary>A loaded Oasis pipeline plus the loaders that own its mmap-backed weights.</summary>
    private sealed class LoadedWorld : IDisposable
    {
        private readonly IReadOnlyList<IDisposable> _owned;

        /// <summary>Wraps <paramref name="pipeline"/> and the loaders whose lifetime it depends on.</summary>
        public LoadedWorld(OasisPipeline pipeline, IReadOnlyList<IDisposable> owned)
        {
            Pipeline = pipeline;
            _owned = owned;
        }

        /// <summary>The loaded pipeline, owned by the service.</summary>
        public OasisPipeline Pipeline { get; }

        /// <inheritdoc/>
        public void Dispose()
        {
            Pipeline.Dispose();
            foreach (IDisposable owned in _owned)
            {
                owned.Dispose();
            }
        }
    }
}
