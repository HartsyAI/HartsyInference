using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Models.CoLa3D;
using HartsyInference.ThreeD.Models.HairDeck;
using HartsyInference.ThreeD.Models.HairDeck.Preprocessing;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>3D service. Image models (TripoSR, Hunyuan3D, TRELLIS) turn an image into GLB (mesh) or PLY (splat); the scene model
/// (CoLa3D) turns a triangle scene into one GLB mesh per decomposed object, and the rigging model (UniRig) turns a triangle mesh into a
/// rigged GLB. Loaded pipelines are cached per checkpoint for the
/// life of the service, and a model id without a branch here is rejected before anything loads.</summary>
public sealed class MeshService : IMeshService, IDisposable
{
    private const string HairDeckId = "hairdeck";

    private readonly InferenceEngine _engine;
    private readonly Dictionary<string, LoadedMesh> _pipelines = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, CoLa3DPipeline> _scenePipelines = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, UniRigPipeline> _rigPipelines = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, HairDeckRuntime> _hairDecks = new(StringComparer.OrdinalIgnoreCase);
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
        if (MeshModelIds.RigMesh.Contains(id))
            return GenerateRiggedAsync(spec, id, request, progress, cancel);

        if (request.ScenePrompt is not null)
            throw new NotSupportedException($"3D model '{id}' takes no scene prompt; ScenePrompt applies to scene models (cola3d).");
        if (request.Image is null)
        {
            throw new NotSupportedException(
                "Text-to-3D has no wired pipeline; supply MeshRequest.Image for image-to-3D (TripoSR / Hunyuan3D).");
        }
        if (IsHairDeck(spec))
        {
            return Task.Run(() => new MeshResult { Data = GenerateHairDeck(spec, request, cancel), Format = "glb" }, cancel);
        }
        HairDeckRequestMapper.RejectForOtherModels(request, id);
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
        HairDeckRequestMapper.RejectForOtherModels(request, id);
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

    /// <summary>Rigging: one GLB holding the input mesh, a generated skeleton and per-vertex skin weights.</summary>
    private Task<MeshResult> GenerateRiggedAsync(ModelSpec spec, string id, MeshRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        HairDeckRequestMapper.RejectForOtherModels(request, id);
        if (request.SceneMesh is null)
        {
            throw new NotSupportedException(
                $"3D model '{id}' rigs a triangle mesh; supply MeshRequest.SceneMesh as obj or glb.");
        }
        if (request.Image is not null)
            throw new NotSupportedException($"3D model '{id}' takes a mesh, not an image; leave MeshRequest.Image empty.");
        if (request.ScenePrompt is not null)
            throw new NotSupportedException($"3D model '{id}' takes no scene prompt; ScenePrompt applies to cola3d.");
        if (request.Steps != 0 || request.GridResolution != 0)
            throw new NotSupportedException($"3D model '{id}' has no steps or grid setting; leave Steps and GridResolution at 0.");
        Mesh input = SceneMeshReader.Read(request.SceneMesh.Data, request.SceneMesh.Format);
        UniRigPipeline.ValidateMesh(input);

        return Task.Run(
            () =>
            {
                cancel.ThrowIfCancellationRequested();
                UniRigPipeline pipeline = GetOrLoadRig(spec, id);
                UniRigResult result = pipeline.Generate(input, request.Seed < 0 ? null : (int)request.Seed, p =>
                {
                    cancel.ThrowIfCancellationRequested();
                    progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
                }, cancel);
                return new MeshResult { Data = result.Glb, Format = "glb" };
            },
            cancel);
    }

    /// <summary>Returns the cached rigging pipeline for <paramref name="spec"/>; the checkpoints themselves load per request.</summary>
    private UniRigPipeline GetOrLoadRig(ModelSpec spec, string id)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                $"No checkpoint found for 3D model '{spec.Requested}'. Supply the folder holding {UniRigPipeline.SkeletonCheckpoint} and {UniRigPipeline.SkinCheckpoint} as the local path.");
        }
        lock (_gate)
        {
            if (_rigPipelines.TryGetValue(spec.LocalPath, out UniRigPipeline? cached))
            {
                return cached;
            }
            try
            {
                UniRigPipeline pipeline = new(_engine.Backend, spec.LocalPath);
                _rigPipelines[spec.LocalPath] = pipeline;
                return pipeline;
            }
            catch (Exception ex)
            {
                Logs.Error($"Failed to prepare the '{id}' 3D pipeline from {spec.LocalPath}", ex);
                throw;
            }
        }
    }

    private static bool IsHairDeck(ModelSpec spec) =>
        string.Equals(spec.Catalog?.Id, HairDeckId, StringComparison.OrdinalIgnoreCase);

    /// <summary>One HairDeck request: the preprocessing of the image, then the image model; the GLB carries the chains when they were made.</summary>
    private byte[] GenerateHairDeck(ModelSpec spec, MeshRequest request, CancellationToken cancel)
    {
        HairDeckOptions options = HairDeckRequestMapper.ToOptions(request);
        HairDeckRuntime runtime = GetOrLoadHairDeck(spec);
        // One run at a time per loaded model: the Vision models and the image model are not shared-safe, and a disposal
        // waits for the run in progress. A queued request gives up when its own cancellation fires.
        runtime.Gate.Wait(cancel);
        try
        {
            if (runtime.IsDisposed)
            {
                throw new ModelUnloadedException("The HairDeck model was unloaded while this request was queued; retry the request.");
            }
            return RunHairDeck(runtime, options, request, cancel);
        }
        finally
        {
            runtime.Gate.Release();
        }
    }

    private byte[] RunHairDeck(HairDeckRuntime runtime, HairDeckOptions options, MeshRequest request, CancellationToken cancel)
    {
        HairDeckSourceImage source = HairDeckSourceImage.Create(request.Image!.Rgb, request.Image.Alpha, request.Image.Width, request.Image.Height);
        // A request cancelled while it queued for the gate must not start SAM 3 and BiRefNet.
        cancel.ThrowIfCancellationRequested();
        HairDeckPreprocessResult prep = runtime.Preprocessor.Process(source);
        foreach (string note in prep.Notes)
        {
            // input_alpha only records that the input's alpha was applied as a mask; it does not degrade the result.
            if (note == "input_alpha")
            {
                Logs.Info("HairDeck: input alpha applied as a mask.");
            }
            else
            {
                Logs.Warning($"HairDeck: {note}");
            }
        }

        cancel.ThrowIfCancellationRequested();
        HairDeckResult result = runtime.Model.Generate(prep.Input, options, cancel);
        if (!result.HasCards)
        {
            throw new InvalidOperationException("No hair cards were generated for this image. Try another seed or another image.");
        }

        if (options.Chains && !result.HasChains)
        {
            throw new InvalidOperationException("Chains were requested but the chain model produced none; no partial result is returned.");
        }

        if (options.Rigged)
        {
            return HairDeckGlbExport.WriteRigged(result, prep.Input.HairBaseColorSrgb);
        }

        return HairDeckGlbExport.Write(result, prep.Input.HairBaseColorSrgb, options.Chains);
    }

    /// <summary>Loads (or returns the cached) HairDeck stack for <paramref name="spec"/>: the model directory is the one holding its config.json.</summary>
    private HairDeckRuntime GetOrLoadHairDeck(ModelSpec spec)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                "No checkpoint found for 'hairdeck'. Download the catalog assets or supply the model directory as the local path.");
        }

        string modelRoot = HairDeckModel.ResolveModelRoot(spec.LocalPath);
        string visionRoot = Path.Combine(RepoPaths.ModelsRoot(), "Vision");
        string key = $"{modelRoot}|{visionRoot}";
        lock (_gate)
        {
            if (_hairDecks.TryGetValue(key, out HairDeckRuntime? cached))
            {
                return cached;
            }

            HairDeckRuntime runtime = HairDeckRuntime.Load(_engine.Backend, modelRoot, visionRoot);
            _hairDecks[key] = runtime;
            return runtime;
        }
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
                    $"3D model '{id}' has no wired image pipeline; supported: {string.Join(", ", MeshModelIds.All.Append(HairDeckId))}.");
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
        for (int i = 0; i < masks.Count; i++)
        {
            if (masks[i] is null || masks[i].Coverage is null)
                throw new ArgumentException($"Mask {i} has no coverage bytes.", nameof(masks));
        }
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
        // Take the entries out under the lock, then dispose outside it: a HairDeck disposal waits for a run in progress,
        // and holding the lock meanwhile would stall every other model load.
        List<LoadedMesh> pipelines;
        List<CoLa3DPipeline> scenePipelines;
        List<UniRigPipeline> rigPipelines;
        List<HairDeckRuntime> hairDecks;
        lock (_gate)
        {
            pipelines = _pipelines.Values.ToList();
            scenePipelines = _scenePipelines.Values.ToList();
            rigPipelines = _rigPipelines.Values.ToList();
            hairDecks = _hairDecks.Values.ToList();
            _pipelines.Clear();
            _scenePipelines.Clear();
            _rigPipelines.Clear();
            _hairDecks.Clear();
        }
        foreach (LoadedMesh loaded in pipelines)
        {
            loaded.Pipeline.Dispose();
        }
        foreach (CoLa3DPipeline pipeline in scenePipelines)
        {
            pipeline.Dispose();
        }
        foreach (UniRigPipeline pipeline in rigPipelines)
        {
            pipeline.Dispose();
        }
        foreach (HairDeckRuntime runtime in hairDecks)
        {
            runtime.Dispose();
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
