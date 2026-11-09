
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Models.CoLa3D;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>3D service. Image models (TripoSR, Hunyuan3D, TRELLIS) turn an image into GLB (mesh) or PLY (splat); the scene model
/// (CoLa3D) turns a triangle scene into one GLB mesh per decomposed object. Loaded pipelines are cached per checkpoint for the
/// life of the service, and a model id without a branch here is rejected before anything loads.</summary>
public sealed class MeshService : IMeshService, IDisposable
{
    private readonly InferenceEngine _engine;
    private readonly Dictionary<string, LoadedMesh> _pipelines = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, CoLa3DPipeline> _scenePipelines = new(StringComparer.OrdinalIgnoreCase);
    private readonly object _gate = new object();

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal MeshService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<MeshResult> GenerateAsync(ModelSpec spec, MeshRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(request);
        string id = (spec.Catalog?.Id ?? spec.Requested ?? string.Empty).ToLowerInvariant();
        if (MeshModelIds.SceneMesh.Contains(id))
            return GenerateSceneAsync(spec, id, request, progress, cancel);

        if (request.ScenePrompt is not null)
            throw new NotSupportedException($"3D model '{id}' takes no scene prompt; ScenePrompt applies to scene models (cola3d).");
        if (request.Image is null)
        {
            throw new NotSupportedException(
                "Text-to-3D has no wired pipeline; supply MeshRequest.Image for image-to-3D (TripoSR / Hunyuan3D).");
        }
        return Task.Run(
            () =>
            {
                cancel.ThrowIfCancellationRequested();
                LoadedMesh loaded = GetOrLoad(spec, id);
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
                if (result.Mesh is { TriangleCount: > 0 } mesh)
                {
                    return new MeshResult { Data = GlbWriter.Write(mesh), Format = "glb" };
                }
                if (result.Splats is { Count: > 0 } splats)
                {
                    return new MeshResult { Data = PlyWriter.WriteSplats(splats), Format = "ply" };
                }
                throw new InvalidOperationException("The pipeline produced no geometry (try a foreground-on-gray input).");
            },
            cancel);
    }

    /// <summary>Scene decomposition: one GLB holding a mesh node per object.</summary>
    private Task<MeshResult> GenerateSceneAsync(ModelSpec spec, string id, MeshRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        if (request.SceneMesh is null)
        {
            throw new NotSupportedException(
                $"3D model '{id}' decomposes a triangle scene; supply MeshRequest.SceneMesh as obj or glb.");
        }
        if (request.Image is not null)
            throw new NotSupportedException($"3D model '{id}' takes a scene mesh, not an image; leave MeshRequest.Image empty.");
        SceneMeshData sceneMesh = request.SceneMesh;
        CoLa3DPrompt? prompt = request.ScenePrompt is null ? null : ToCoLa3DPrompt(request.ScenePrompt);
        CoLa3DConfig.QuerySourceKind source = prompt?.Kind ?? CoLa3DConfig.QuerySourceKind.Fixed;
        // Rejects the request before the checkpoint loads; the pipeline checks it again on Generate.
        CoLa3DPipeline.ValidateRequest(CoLa3DConfig.ForQuerySource(source), request.GridResolution, prompt);

        return Task.Run(
            () =>
            {
                cancel.ThrowIfCancellationRequested();
                CoLa3DPipeline pipeline = GetOrLoadScene(spec, id, source);
                Mesh scene = SceneMeshReader.Read(sceneMesh.Data, sceneMesh.Format);
                CoLa3DResult result = pipeline.Generate(scene, request.GridResolution, p =>
                {
                    cancel.ThrowIfCancellationRequested();
                    progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
                }, prompt);
                if (result.Objects.Count == 0)
                    throw new InvalidOperationException("The scene decomposed into no object geometry (try a finer grid).");
                return new MeshResult { Data = GlbWriter.Write(result.Objects), Format = "glb" };
            },
            cancel);
    }

    /// <summary>Loads (or returns the cached) image-to-3D pipeline for <paramref name="spec"/>.</summary>
    private LoadedMesh GetOrLoad(ModelSpec spec, string id)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                $"No checkpoint found for 3D model '{spec.Requested}'. Supply its model directory or checkpoint "
                + "as the local path.");
        }
        lock (_gate)
        {
            if (_pipelines.TryGetValue(spec.LocalPath, out LoadedMesh? cached))
            {
                return cached;
            }
            if (!MeshModelIds.ImageToMesh.Contains(id))
            {
                throw new NotSupportedException(
                    $"3D model '{id}' has no wired image pipeline; supported: {string.Join(", ", MeshModelIds.All)}.");
            }
            try
            {
                LoadedMesh loaded;
                if (id == "hunyuan3d")
                {
                    Hunyuan3DShapePipeline hunyuan = Hunyuan3DShapePipeline.LoadFromPath(_engine.Backend, spec.LocalPath);
                    loaded = new LoadedMesh(hunyuan, hunyuan.Generate);
                }
                else if (id == "trellis")
                {
                    TrellisImageTo3DPipeline trellis = TrellisImageTo3DPipeline.LoadFromPath(_engine.Backend, spec.LocalPath);
                    loaded = new LoadedMesh(trellis, trellis.Generate);
                }
                else if (id == "triposr")
                {
                    TripoSrPipeline triposr = TripoSrPipeline.LoadFromPath(_engine.Backend, spec.LocalPath);
                    loaded = new LoadedMesh(triposr, triposr.Generate);
                }
                else
                {
                    // Unreachable while MeshModelIds.ImageToMesh matches these branches; a new id must get its own branch.
                    throw new System.Diagnostics.UnreachableException($"No loader wired for image 3D model '{id}'.");
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

    /// <summary>Converts the wire prompt to the pipeline's prompt, checking that exactly one kind is given.</summary>
    private static CoLa3DPrompt ToCoLa3DPrompt(SceneMeshPrompt prompt)
    {
        if ((prompt.Masks is null) == (prompt.Points is null))
            throw new ArgumentException("A scene prompt gives masks or points, not both and not neither.", nameof(prompt));
        return prompt.Masks is null ? PointPrompt(prompt.Points!) : MaskPrompt(prompt.Masks);
    }

    private static CoLa3DPrompt PointPrompt(IReadOnlyList<ScenePoint> points)
    {
        float[] flat = new float[points.Count * 3];
        for (int i = 0; i < points.Count; i++)
        {
            flat[i * 3] = points[i].X;
            flat[i * 3 + 1] = points[i].Y;
            flat[i * 3 + 2] = points[i].Z;
        }

        return new CoLa3DPrompt { Queries = points.Count, Points = flat };
    }

    private static CoLa3DPrompt MaskPrompt(IReadOnlyList<ScenePromptMask> masks)
    {
        if (masks.Count == 0)
            throw new ArgumentException("A mask prompt names at least one object.", nameof(masks));
        int width = masks[0].Width, height = masks[0].Height;
        // Multiply in long: a plane that wraps an int could pass the length checks and then allocate a huge tensor later.
        long planeLong = (long)width * height;
        if (planeLong < 0 || planeLong > int.MaxValue / masks.Count)
        {
            throw new ArgumentException(
                $"Mask prompt sizes are out of range: {masks.Count} masks of {width}x{height} must fit one {int.MaxValue}-byte buffer.",
                nameof(masks));
        }

        int plane = (int)planeLong;
        byte[] coverage = new byte[masks.Count * plane];
        for (int i = 0; i < masks.Count; i++)
        {
            ScenePromptMask mask = masks[i];
            if (mask.Width != width || mask.Height != height || mask.Coverage.Length != plane)
                throw new ArgumentException($"Mask {i} must be {width}x{height} with {plane} coverage bytes.", nameof(masks));
            mask.Coverage.CopyTo(coverage, i * plane);
        }

        return new CoLa3DPrompt { Queries = masks.Count, MaskCoverage = coverage, MaskHeight = height, MaskWidth = width };
    }

    /// <summary>Loads (or returns the cached) scene pipeline for <paramref name="spec"/> and decomposer variant.</summary>
    private CoLa3DPipeline GetOrLoadScene(ModelSpec spec, string id, CoLa3DConfig.QuerySourceKind source)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                $"No checkpoint found for 3D model '{spec.Requested}'. Supply its model directory as the local path.");
        }
        string key = $"{spec.LocalPath}|{source}";
        lock (_gate)
        {
            if (_scenePipelines.TryGetValue(key, out CoLa3DPipeline? cached))
            {
                return cached;
            }
            if (id != "cola3d")
            {
                throw new System.Diagnostics.UnreachableException($"No loader wired for scene 3D model '{id}'.");
            }
            try
            {
                CoLa3DPipeline pipeline = CoLa3DPipeline.LoadFromPath(_engine.Backend, spec.LocalPath, source);
                _scenePipelines[key] = pipeline;
                return pipeline;
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
            foreach (CoLa3DPipeline pipeline in _scenePipelines.Values)
            {
                pipeline.Dispose();
            }
            _scenePipelines.Clear();
        }
    }

    /// <summary>A loaded image-to-3D pipeline plus its uniform generate entry point, so the service does not branch on the concrete pipeline type per request.</summary>
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
