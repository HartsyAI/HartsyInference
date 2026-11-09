using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Models.HairDeck;
using HartsyInference.ThreeD.Models.HairDeck.Preprocessing;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Image-to-3D service: loads TripoSR (feed-forward), Hunyuan3D (flow-match, mesh), or TRELLIS (two-stage flow, Gaussian splat) from the spec's model directory, runs one image-to-3D pass, and returns the geometry encoded as GLB (mesh models) or binary-PLY (splat models — TRELLIS's flexicubes mesh decoder is not yet ported, so it only ever emits splats). Loaded pipelines are cached per checkpoint for the life of the service; text-to-3D has no wired pipeline yet, so a request without an image is rejected by name.</summary>
public sealed class MeshService : IMeshService, IDisposable
{
    private const string HairDeckId = "hairdeck";
    private readonly InferenceEngine _engine;
    private readonly Dictionary<string, LoadedMesh> _pipelines = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, HairDeckRuntime> _hairDecks = new(StringComparer.OrdinalIgnoreCase);
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
        if (IsHairDeck(spec))
        {
            return Task.Run(() => new MeshResult { Data = GenerateHairDeck(spec, request, cancel), Format = "glb" }, cancel);
        }
        HairDeckRequestMapper.RejectForOtherModels(request, spec.Catalog?.Id ?? spec.Requested);
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
                throw new ObjectDisposedException(nameof(HairDeckRuntime), "The HairDeck model was unloaded while this request was queued.");
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
    private LoadedMesh GetOrLoad(ModelSpec spec)
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
            string id = (spec.Catalog?.Id ?? "triposr").ToLowerInvariant();
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
        // Take the entries out under the lock, then dispose outside it: a HairDeck disposal waits for a run in progress,
        // and holding the lock meanwhile would stall every other model load.
        List<LoadedMesh> pipelines;
        List<HairDeckRuntime> hairDecks;
        lock (_gate)
        {
            pipelines = _pipelines.Values.ToList();
            hairDecks = _hairDecks.Values.ToList();
            _pipelines.Clear();
            _hairDecks.Clear();
        }
        foreach (LoadedMesh loaded in pipelines)
        {
            loaded.Pipeline.Dispose();
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
