using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Image-to-3D service: loads TripoSR (feed-forward) or Hunyuan3D (flow-match) from the spec's model
/// directory, runs one image-to-mesh pass, and returns the mesh encoded as GLB. Loaded pipelines are cached per
/// checkpoint for the life of the service; text-to-3D has no wired pipeline yet, so a request without an image is
/// rejected by name.</summary>
public sealed class MeshService : IMeshService, IDisposable
{
    private readonly InferenceEngine _engine;
    private readonly Dictionary<string, LoadedMesh> _pipelines = new(StringComparer.OrdinalIgnoreCase);
    private readonly object _gate = new object();

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal MeshService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<MeshResult> GenerateAsync(ModelSpec spec, MeshRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(request);
        if (request.Image is null)
        {
            throw new NotSupportedException(
                "Text-to-3D has no wired pipeline; supply MeshRequest.Image for image-to-3D (TripoSR / Hunyuan3D).");
        }
        return Task.Run(
            () =>
            {
                cancel.ThrowIfCancellationRequested();
                LoadedMesh loaded = GetOrLoad(spec);
                ImageTo3DRequest job = new ImageTo3DRequest
                {
                    ImageRgb = request.Image.Rgb,
                    Width = request.Image.Width,
                    Height = request.Image.Height,
                    GridResolution = request.GridResolution,
                    Steps = request.Steps,
                    Seed = request.Seed < 0 ? null : (int)request.Seed,
                };
                ThreeDResult result = loaded.Generate(job, p =>
                {
                    cancel.ThrowIfCancellationRequested();
                    progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
                });
                if (result.Mesh is not { TriangleCount: > 0 } mesh)
                {
                    throw new InvalidOperationException("The pipeline produced an empty mesh (try a foreground-on-gray input).");
                }
                return new MeshResult { Data = GlbWriter.Write(mesh), Format = "glb" };
            },
            cancel);
    }

    /// <summary>Loads (or returns the cached) image-to-3D pipeline for <paramref name="spec"/>.</summary>
    private LoadedMesh GetOrLoad(ModelSpec spec)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException("No 3D model found. Pass the model directory (or checkpoint) as the spec's local path.");
        }
        lock (_gate)
        {
            if (_pipelines.TryGetValue(spec.LocalPath, out LoadedMesh? cached))
            {
                return cached;
            }
            string id = (spec.Catalog?.Id ?? "triposr").ToLowerInvariant();
            try
            {
                LoadedMesh loaded;
                if (id == "hunyuan3d")
                {
                    Hunyuan3DShapePipeline hunyuan = Hunyuan3DShapePipeline.LoadFromPath(_engine.Backend, spec.LocalPath);
                    loaded = new LoadedMesh(hunyuan, hunyuan.Generate);
                }
                else
                {
                    TripoSrPipeline triposr = TripoSrPipeline.LoadFromPath(_engine.Backend, spec.LocalPath);
                    loaded = new LoadedMesh(triposr, triposr.Generate);
                }
                _pipelines[spec.LocalPath] = loaded;
                return loaded;
            }
            catch (Exception ex)
            {
                Logs.Error($"Failed to load the '{id}' 3D pipeline from {spec.LocalPath}", ex);
                throw;
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_gate)
        {
            foreach (LoadedMesh loaded in _pipelines.Values)
            {
                loaded.Pipeline.Dispose();
            }
            _pipelines.Clear();
        }
    }

    /// <summary>A loaded image-to-3D pipeline plus its uniform generate entry point, so the service does not branch
    /// on the concrete pipeline type per request.</summary>
    private sealed class LoadedMesh
    {
        /// <summary>Wraps <paramref name="pipeline"/> with its <paramref name="generate"/> entry point.</summary>
        public LoadedMesh(IDisposable pipeline, Func<ImageTo3DRequest, Action<GenerationProgress>?, ThreeDResult> generate)
        {
            Pipeline = pipeline;
            Generate = generate;
        }

        /// <summary>The loaded pipeline, owned by the service.</summary>
        public IDisposable Pipeline { get; }

        /// <summary>Runs one image-to-mesh request.</summary>
        public Func<ImageTo3DRequest, Action<GenerationProgress>?, ThreeDResult> Generate { get; }
    }
}
