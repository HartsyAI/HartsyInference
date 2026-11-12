using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Models.CoLa3D;
using HartsyInference.ThreeD.Models.HairDeck;
using HartsyInference.ThreeD.Models.HairDeck.Preprocessing;
using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;
using HartsyInference.ThreeD.Pipelines;
using HartsyInference.ThreeD.Pipelines.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>3D service. Image models (TripoSR, Hunyuan3D, TRELLIS, TRELLIS.2, which can also texture, Pixal3D) turn an image into GLB (mesh) or PLY (splat); the scene model
/// (CoLa3D) turns a triangle scene into one GLB mesh per decomposed object, and the rigging model (UniRig) turns a triangle mesh into a
/// rigged GLB. Loaded pipelines are cached per checkpoint for the
/// life of the service, and a model id without a branch here is rejected before anything loads.</summary>
public sealed class MeshService : IMeshService, IMeshLifecycleService, IDisposable
{
    private static readonly string EngineVersion = typeof(MeshService).Assembly.GetCustomAttributes(typeof(System.Reflection.AssemblyInformationalVersionAttribute), false)
        .OfType<System.Reflection.AssemblyInformationalVersionAttribute>().FirstOrDefault()?.InformationalVersion
        ?? typeof(MeshService).Assembly.GetName().Version?.ToString() ?? "unknown";

    private readonly InferenceEngine _engine;
    private readonly Dictionary<string, LoadedMesh> _pipelines = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, CoLa3DPipeline> _scenePipelines = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, UniRigPipeline> _rigPipelines = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, HunyuanPaintLoader> _paintPipelines = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, P3SamPipeline> _partPipelines = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, XPartPipeline> _xpartPipelines = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, HairDeckRuntime> _hairDecks = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, Trellis2Runtime> _trellis2 = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, Pixal3DRuntime> _pixal3d = new(StringComparer.OrdinalIgnoreCase);
    private readonly object _gate = new object();
    private readonly SemaphoreSlim _sam3dGate = new(1, 1);
    private int _disposals;

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal MeshService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public async Task<MeshResult> GenerateAsync(ModelSpec spec, MeshRequest request, IProgress<StepPreview>? progress = null, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(request);
        string id = IdOf(spec);
        MeshSettingsUsed settings = SettingsOf(id, request);
        MeshRequest effective = settings.Seed is long seed && request.Seed < 0 ? request with { Seed = seed } : request;
        MeshResult raw = await WithOutOfVramRetry(
            () => GenerateRawAsync(id, spec, effective, progress, cancel),
            () =>
            {
                progress?.Report(MeshProgress.Plain(0, 0, "Retrying after out of memory"));
                FreeAfterOutOfVram();
            }).ConfigureAwait(false);
        return await Task.Run(() => MeshResultFinisher.Finish(id, request, raw, settings, cancel), cancel).ConfigureAwait(false);
    }

    /// <summary>Runs <paramref name="run"/>; when it fails for lack of device memory, calls <paramref name="free"/> and runs it once more. The retry re-resolves its pipeline, which is why it can live here: the failed run's cached pipeline was freed, so nothing believes weights it no longer has are resident. A second failure, any other failure and a cancellation propagate.</summary>
    internal static async Task<MeshResult> WithOutOfVramRetry(Func<Task<MeshResult>> run, Action free)
    {
        try
        {
            return await run().ConfigureAwait(false);
        }
        catch (Exception ex) when (InferenceEngine.IsOutOfVram(ex))
        {
            Logs.Warning($"[Mesh] A generation ran out of VRAM; freeing the loaded 3D models and retrying once: {ex.Message}");
            free();
            return await run().ConfigureAwait(false);
        }
    }

    /// <summary>Gives back what the 3D side holds: its resident pipelines and the matting model, then the driver pool's cached-but-free blocks. It does not sweep the backend's weight and activation caches, which the engine's cached image and video pipelines still believe are resident (a sweep under them returns garbage, see <c>InferenceEngine.GenerateWithVramCleanup</c>).</summary>
    private void FreeAfterOutOfVram()
    {
        Free();
        _engine.MeshPreparationInternal.Dispose();
        try
        {
            _engine.Backend.TrimMemoryPool();
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            Logs.Error("[Mesh] Trimming the device pool after an out-of-memory failure failed; the retry may fail too.", ex);
        }
    }

    /// <summary>The pipelines that have no run gate of their own: a run takes a count, and a disposal waits for the count to reach zero and holds new runs off meanwhile.</summary>
    private static bool HasNoRunGate(string kind) => kind is "image" or "scene" or "rig" or "paint" or "part";

    /// <summary>Whether the model's runtime serialises its own runs (and so waits for one before it is disposed); every other model runs counted.</summary>
    private static bool RunsGated(string id, ModelSpec spec) =>
        id == MeshModelIds.Trellis2 || id == MeshModelIds.Pixal3D || id == MeshModelIds.Sam3dObjects || id == MeshModelIds.HairDeck || IsHairDeck(spec);

    private readonly object _runLock = new();
    private int _ungatedRuns;
    private bool _disposing;
    private TaskCompletionSource _reopened = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private TaskCompletionSource? _drained;

    /// <summary>Runs one generation on a pipeline without a run gate, counted so that <see cref="Free"/> cannot dispose it under the run. While a disposal is waiting or running the run waits without holding a thread, and gives up when <paramref name="cancel"/> is cancelled.</summary>
    internal async Task<MeshResult> RunCounted(Func<Task<MeshResult>> run, CancellationToken cancel = default)
    {
        while (true)
        {
            Task reopened;
            lock (_runLock)
            {
                if (!_disposing)
                {
                    _ungatedRuns++;
                    break;
                }
                reopened = _reopened.Task;
            }
            await reopened.WaitAsync(cancel).ConfigureAwait(false);
        }
        try
        {
            return await run().ConfigureAwait(false);
        }
        finally
        {
            lock (_runLock)
            {
                if (--_ungatedRuns == 0)
                    _drained?.TrySetResult();
            }
        }
    }

    /// <summary>Runs <paramref name="dispose"/> once no counted run is in progress, holding new ones off until it finishes. The caller's thread waits: the lifecycle calls it serves are synchronous.</summary>
    private void WhenNoCountedRun(Action dispose)
    {
        TaskCompletionSource drained = new(TaskCreationOptions.RunContinuationsAsynchronously);
        TaskCompletionSource reopened = new(TaskCreationOptions.RunContinuationsAsynchronously);
        while (true)
        {
            Task? other = null;
            lock (_runLock)
            {
                if (_disposing)
                    other = _reopened.Task;
                else
                {
                    _disposing = true;
                    _reopened = reopened;
                    _drained = drained;
                    if (_ungatedRuns == 0)
                        drained.TrySetResult();
                    break;
                }
            }
            other.Wait();
        }
        try
        {
            drained.Task.Wait();
            dispose();
        }
        finally
        {
            lock (_runLock)
            {
                _disposing = false;
                _drained = null;
            }
            reopened.TrySetResult();
        }
    }

    /// <inheritdoc/>
    public IReadOnlyList<MeshLoadedModel> Loaded()
    {
        lock (_gate)
        {
            return [.. Resident().Select(entry => new MeshLoadedModel(entry.Kind, NameOf(entry.Key)))];
        }
    }

    /// <inheritdoc/>
    public int Free(string? name = null)
    {
        List<(string Kind, string Name, IDisposable Item)> freed = [];
        lock (_gate)
        {
            foreach ((string kind, string key, IDisposable item) in Resident().ToList())
            {
                if (name is not null && !string.Equals(NameOf(key), name, StringComparison.OrdinalIgnoreCase))
                    continue;
                RemoveResident(kind, key);
                freed.Add((kind, NameOf(key), item));
            }
        }
        // A disposal waits for the run in progress, so it happens outside the lock that every load takes: the gated runtimes wait on their own gate, the others on the run count.
        foreach ((string kind, string _, IDisposable item) in freed.Where(f => !HasNoRunGate(f.Kind)))
            item.Dispose();
        List<IDisposable> counted = [.. freed.Where(f => HasNoRunGate(f.Kind)).Select(f => f.Item)];
        if (counted.Count > 0)
            WhenNoCountedRun(() => counted.ForEach(item => item.Dispose()));
        if (freed.Count > 0)
            Logs.Info($"[Mesh] Freed {freed.Count} resident 3D model(s): {string.Join(", ", freed.Select(f => $"{f.Kind} {f.Name}"))}.");
        return freed.Count;
    }

    private IEnumerable<(string Kind, string Key, IDisposable Item)> Resident()
    {
        foreach (KeyValuePair<string, LoadedMesh> e in _pipelines) yield return ("image", e.Key, e.Value.Pipeline);
        foreach (KeyValuePair<string, CoLa3DPipeline> e in _scenePipelines) yield return ("scene", e.Key, e.Value);
        foreach (KeyValuePair<string, UniRigPipeline> e in _rigPipelines) yield return ("rig", e.Key, e.Value);
        foreach (KeyValuePair<string, HunyuanPaintLoader> e in _paintPipelines) yield return ("paint", e.Key, e.Value);
        foreach (KeyValuePair<string, P3SamPipeline> e in _partPipelines) yield return ("part", e.Key, e.Value);
        foreach (KeyValuePair<string, HairDeckRuntime> e in _hairDecks) yield return ("hairdeck", e.Key, e.Value);
        foreach (KeyValuePair<string, Trellis2Runtime> e in _trellis2) yield return ("trellis2", e.Key, e.Value);
        foreach (KeyValuePair<string, Pixal3DRuntime> e in _pixal3d) yield return ("pixal3d", e.Key, e.Value);
    }

    private void RemoveResident(string kind, string key)
    {
        switch (kind)
        {
            case "image": _pipelines.Remove(key); break;
            case "scene": _scenePipelines.Remove(key); break;
            case "rig": _rigPipelines.Remove(key); break;
            case "paint": _paintPipelines.Remove(key); break;
            case "part": _partPipelines.Remove(key); break;
            case "hairdeck": _hairDecks.Remove(key); break;
            case "trellis2": _trellis2.Remove(key); break;
            default: _pixal3d.Remove(key); break;
        }
    }

    /// <summary>The key a pipeline is cached under as a name that holds no absolute path: relative to the models root when under it, else the last segment. A scene key carries its variant after a bar and keeps it.</summary>
    private static string NameOf(string key)
    {
        string[] parts = key.Split('|');
        string root = RepoPaths.ModelsRoot();
        string path = parts[0];
        string name = path.StartsWith(root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)
            ? Path.GetRelativePath(root, path).Replace('\\', '/')
            : Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        return parts.Length > 1 ? name + "|" + parts[1] : name;
    }

    /// <summary>The settings a request runs with: the planned values and sources, and the seed resolved now (a random draw for a negative request) so the result can report and embed it. Unknown models fail later, in validation.</summary>
    internal static MeshSettingsUsed SettingsOf(string id, MeshRequest request)
    {
        if (MeshModelProfiles.Find(id) is null)
            return new MeshSettingsUsed { ModelId = id, EngineVersion = EngineVersion };
        MeshModelDescriptor descriptor = MeshPlanningService.DescriptorOf(id);
        Dictionary<string, ResolvedParameter> parameters = new(MeshPlanningService.Settings(request, descriptor));
        long? seed = null;
        if (descriptor.Fields[nameof(MeshRequest.Seed)].Access == MeshFieldAccess.Used)
        {
            seed = MeshSeed.Resolve(request.Seed);
            parameters[nameof(MeshRequest.Seed)] = ParameterProvenance.Seed(request.Seed, seed.Value);
        }
        return new MeshSettingsUsed { ModelId = id, Seed = seed, Parameters = parameters, EngineVersion = EngineVersion };
    }

    private Task<MeshResult> GenerateRawAsync(string id, ModelSpec spec, MeshRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        MeshRequestValidator.Validate(id, request, MatterAvailable());
        return RunsGated(id, spec) ? DispatchAsync(id, spec, request, progress, cancel) : RunCounted(() => DispatchAsync(id, spec, request, progress, cancel), cancel);
    }

    private Task<MeshResult> DispatchAsync(string id, ModelSpec spec, MeshRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        if (MeshModelIds.SceneMesh.Contains(id))
            return GenerateSceneAsync(spec, id, request, progress, cancel);
        if (MeshModelIds.RigMesh.Contains(id))
            return GenerateRiggedAsync(spec, id, request, progress, cancel);
        if (MeshModelIds.PaintMesh.Contains(id))
            return GeneratePaintAsync(spec, id, request, progress, cancel);
        if (MeshModelIds.PartMesh.Contains(id))
            return GeneratePartAsync(spec, id, request, progress, cancel);
        if (MeshModelIds.XPartMesh.Contains(id))
            return GenerateXPartAsync(spec, id, request, progress, cancel);

        if (IsHairDeck(spec))
        {
            return Task.Run(() => new MeshResult { Data = GenerateHairDeck(spec, request, progress, cancel), Format = "glb" }, cancel);
        }
        if (id == MeshModelIds.Trellis2)
            return GenerateTrellis2Async(spec, request, progress, cancel);
        if (id == MeshModelIds.Pixal3D)
            return GeneratePixal3DAsync(spec, request, progress, cancel);
        if (id == MeshModelIds.Sam3dObjects)
            return GenerateSam3dAsync(spec, request, progress, cancel);
        return Task.Run(
            () =>
            {
                cancel.ThrowIfCancellationRequested();
                // Matting first, so on a cold start the matting forward pass does not overlap the shape model's first load; the matting weights stay resident afterwards.
                ImageData image = request.Image!;
                if (request.Preprocess == MeshPreprocess.Auto)
                    image = _engine.MeshPreparationInternal.Prepare(id, image, cancel).Image;
                LoadedMesh loaded = GetOrLoad(spec, id);
                ImageTo3DRequest job = new ImageTo3DRequest
                {
                    ImageRgb = image.Rgb,
                    Width = image.Width,
                    Height = image.Height,
                    GridResolution = request.GridResolution,
                    Steps = request.Steps,
                    Seed = MeshSeed.ForSampler(request.Seed),
                };
                ThreeDResult result = loaded.Generate(job, p =>
                {
                    cancel.ThrowIfCancellationRequested();
                    progress?.Report(MeshProgress.Plain(p.Step, p.TotalSteps));
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
        SceneMeshData sceneMesh = request.SceneMesh!;
        CoLa3DPrompt? prompt = request.ScenePrompt is null ? null : ToCoLa3DPrompt(request.ScenePrompt);
        CoLa3DConfig.QuerySourceKind source = prompt?.Kind ?? CoLa3DConfig.QuerySourceKind.Fixed;

        return Task.Run(
            () =>
            {
                cancel.ThrowIfCancellationRequested();
                CoLa3DPipeline pipeline = GetOrLoadScene(spec, id, source);
                Mesh scene = SceneMeshReader.Read(sceneMesh.Data, sceneMesh.Format);
                CoLa3DResult result = pipeline.Generate(scene, request.GridResolution, p =>
                {
                    cancel.ThrowIfCancellationRequested();
                    progress?.Report(MeshProgress.Plain(p.Step, p.TotalSteps));
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
        Mesh input = SceneMeshReader.Read(request.SceneMesh!.Data, request.SceneMesh.Format);
        UniRigPipeline.ValidateMesh(input);

        return Task.Run(
            () =>
            {
                cancel.ThrowIfCancellationRequested();
                UniRigPipeline pipeline = GetOrLoadRig(spec, id);
                UniRigResult result = pipeline.Generate(input, MeshSeed.ForSampler(request.Seed), p =>
                {
                    cancel.ThrowIfCancellationRequested();
                    progress?.Report(MeshProgress.Plain(p.Step, p.TotalSteps));
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

    /// <summary>Part segmentation: one GLB holding the cleaned input mesh split into one coloured node per part, and the parts' labels and boxes in
    /// <see cref="MeshResult.Parts"/>. <see cref="MeshRequest.Seed"/> fixes the surface sampling; the point and prompt counts are the upstream ones (100,000 and 400).</summary>
    private Task<MeshResult> GeneratePartAsync(ModelSpec spec, string id, MeshRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        Mesh input = SceneMeshReader.Read(request.SceneMesh!.Data, request.SceneMesh.Format);
        P3SamPipeline.ValidateMesh(input);

        return Task.Run(
            () =>
            {
                cancel.ThrowIfCancellationRequested();
                P3SamPipeline pipeline = GetOrLoadPart(spec, id);
                P3SamResult result = pipeline.Generate(input, MeshSeed.ForSampler(request.Seed), onProgress: p =>
                {
                    cancel.ThrowIfCancellationRequested();
                    progress?.Report(MeshProgress.Plain(p.Step, p.TotalSteps));
                }, cancel: cancel);
                MeshPart[] parts = [.. result.PartIds.Select((partId, i) => new MeshPart(partId, result.Boxes[i], result.FaceCounts[i]))];
                return new MeshResult { Data = result.Glb, Format = "glb", Parts = parts };
            },
            cancel);
    }

    /// <summary>X-Part: P3-SAM splits the mesh into parts and every part is generated again as a complete shape. One GLB with a coloured node per part, the parts' boxes and face counts in
    /// <see cref="MeshResult.Parts"/>. <see cref="MeshRequest.Seed"/> fixes the samples and the noise (negative: 42), <see cref="MeshRequest.Steps"/> the flow steps (0: 50) and
    /// <see cref="MeshRequest.GridResolution"/> the marching-cubes grid of each part (0: 256).</summary>
    private Task<MeshResult> GenerateXPartAsync(ModelSpec spec, string id, MeshRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        Mesh input = SceneMeshReader.Read(request.SceneMesh!.Data, request.SceneMesh.Format);
        P3SamPipeline.ValidateMesh(input);

        return Task.Run(
            () =>
            {
                cancel.ThrowIfCancellationRequested();
                XPartPipeline pipeline = GetOrLoadXPart(spec, id);
                XPartOptions options = new()
                {
                    Seed = MeshSeed.ForSampler(request.Seed) ?? 42,
                    Steps = request.Steps == 0 ? 50 : request.Steps,
                    GridResolution = request.GridResolution == 0 ? 256 : request.GridResolution,
                };
                XPartResult result = pipeline.Generate(input, options, p =>
                {
                    cancel.ThrowIfCancellationRequested();
                    progress?.Report(new StepPreview { Step = p.Step, TotalSteps = p.TotalSteps });
                }, cancel);
                // The id is the part's position in the output (X-Part regenerates the parts, so P3-SAM's labels do not apply to them).
                MeshPart[] parts = [.. result.Parts.Select((part, i) => new MeshPart(i, result.Boxes[i], part.TriangleCount))];
                return new MeshResult { Data = result.Glb, Format = "glb", Parts = parts };
            },
            cancel);
    }

    /// <summary>Returns the cached X-Part pipeline for <paramref name="spec"/>: the folder holding the four checkpoints (each stage loads its file when it runs).</summary>
    private XPartPipeline GetOrLoadXPart(ModelSpec spec, string id)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                $"No checkpoint found for 3D model '{spec.Requested}'. Download its catalog assets or supply the folder holding {string.Join(", ", XPartPipeline.Files)} as the local path.");
        }
        string folder = XPartFolder(spec.LocalPath);
        lock (_gate)
        {
            if (_xpartPipelines.TryGetValue(folder, out XPartPipeline? cached))
            {
                return cached;
            }
            try
            {
                XPartPipeline pipeline = new(_engine.Backend, folder);
                _xpartPipelines[folder] = pipeline;
                return pipeline;
            }
            catch (Exception ex)
            {
                Logs.Error($"Failed to prepare the '{id}' 3D pipeline from {spec.LocalPath}", ex);
                throw;
            }
        }
    }

    // The folder holding p3sam/, conditioner/, model/ and shapevae/: the path itself, or the folder above the file or subfolder a path points at.
    internal static string XPartFolder(string localPath)
    {
        string folder = File.Exists(localPath) ? Path.GetDirectoryName(Path.GetFullPath(localPath))! : localPath;
        string? parent = Path.GetDirectoryName(folder.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        foreach (string? candidate in new[] { folder, parent })
        {
            if (candidate is not null && XPartPipeline.Files.All(file => File.Exists(Path.Combine(candidate, file))))
            {
                return Path.GetFullPath(candidate);
            }
        }
        // Name what is missing from the candidate that holds the most of the files (the path may point at a file or a subfolder of the install).
        string best = new[] { folder, parent }.Where(c => c is not null).Select(c => c!).MaxBy(c => XPartPipeline.Files.Count(file => File.Exists(Path.Combine(c, file))))!;
        string missing = string.Join(", ", XPartPipeline.Files.Where(file => !File.Exists(Path.Combine(best, file))));
        throw new FileNotFoundException($"'{localPath}' is missing X-Part checkpoints: {missing}.", Path.Combine(best, XPartPipeline.Files[0]));
    }

    /// <summary>Returns the cached P3-SAM pipeline for <paramref name="spec"/>: <c>p3sam.safetensors</c>, given directly or found in the folder it was given as.</summary>
    private P3SamPipeline GetOrLoadPart(ModelSpec spec, string id)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                $"No checkpoint found for 3D model '{spec.Requested}'. Download its catalog assets or supply {P3SamPipeline.Checkpoint} (or its folder) as the local path.");
        }
        string path = PartCheckpoint(spec.LocalPath);
        lock (_gate)
        {
            if (_partPipelines.TryGetValue(path, out P3SamPipeline? cached))
            {
                return cached;
            }
            try
            {
                P3SamPipeline pipeline = new(_engine.Backend, path);
                _partPipelines[path] = pipeline;
                return pipeline;
            }
            catch (Exception ex)
            {
                Logs.Error($"Failed to prepare the '{id}' 3D pipeline from {spec.LocalPath}", ex);
                throw;
            }
        }
    }

    // The checkpoint file: the path itself when it is a file, else p3sam.safetensors in the folder or in its p3sam/ subfolder.
    internal static string PartCheckpoint(string localPath)
    {
        if (File.Exists(localPath))
        {
            return Path.GetFullPath(localPath);
        }
        foreach (string candidate in new[] { Path.Combine(localPath, P3SamPipeline.Checkpoint), Path.Combine(localPath, "p3sam", P3SamPipeline.Checkpoint) })
        {
            if (File.Exists(candidate))
            {
                return Path.GetFullPath(candidate);
            }
        }
        throw new FileNotFoundException($"No {P3SamPipeline.Checkpoint} in '{localPath}'.", Path.Combine(localPath, P3SamPipeline.Checkpoint));
    }

    /// <summary>Texturing: one GLB holding the input mesh with a baked PBR material painted from the reference image. <see cref="MeshRequest.Steps"/> is the sampler step
    /// count (0 takes the checkpoint default), <see cref="MeshRequest.GridResolution"/> the texture size (0 is 2048) and <see cref="MeshRequest.Seed"/> the noise seed.</summary>
    private Task<MeshResult> GeneratePaintAsync(ModelSpec spec, string id, MeshRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        Mesh input = SceneMeshReader.Read(request.SceneMesh!.Data, request.SceneMesh.Format);
        HunyuanPaintOptions options = new()
        {
            Steps = request.Steps == 0 ? null : request.Steps,
            TextureSize = request.GridResolution == 0 ? 2048 : request.GridResolution,
            Seed = MeshSeed.ForSampler(request.Seed),
        };

        return Task.Run(
            () =>
            {
                cancel.ThrowIfCancellationRequested();
                ImageData reference = request.Image!;
                if (request.Preprocess == MeshPreprocess.Auto)
                    reference = _engine.MeshPreparationInternal.Prepare(id, reference, cancel).Image;
                byte[] rgb = reference.HasAlpha ? CompositeOnWhite(reference) : reference.Rgb;
                HunyuanPaintLoader loader = GetOrLoadPaint(spec, id);
                ThreeD.Geometry.Ops.PbrPostProcessResult result = loader.Pipeline.Generate(input, rgb, reference.Width, reference.Height, options, p =>
                {
                    cancel.ThrowIfCancellationRequested();
                    progress?.Report(MeshProgress.Plain(p.Step, p.TotalSteps));
                }, cancel);
                return new MeshResult { Data = PbrGlbWriter.Write(result.Mesh, result.Material), Format = "glb" };
            },
            cancel);
    }

    /// <summary>The reference composited on white, as the Paint pipelines expect.</summary>
    private static byte[] CompositeOnWhite(ImageData image)
    {
        byte[] result = new byte[image.Rgb.Length];
        for (int i = 0; i < image.Alpha!.Length; i++)
        {
            float a = image.Alpha[i] / 255f;
            for (int c = 0; c < 3; c++)
                result[i * 3 + c] = (byte)Math.Round(image.Rgb[i * 3 + c] * a + 255f * (1f - a));
        }
        return result;
    }

    /// <summary>Returns the cached Paint pipeline for <paramref name="spec"/>: the folder holding <c>unet/</c> and <c>vae/</c> (and, for 2.1, <c>dino/model.safetensors</c>
    /// and, when <c>numerics.paintUpscaleViews</c> is on, <c>RealESRGAN_x4plus.pth</c>); the tensors of the checkpoint decide the generation.</summary>
    private HunyuanPaintLoader GetOrLoadPaint(ModelSpec spec, string id)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                $"No checkpoint found for 3D model '{spec.Requested}'. Download its catalog assets or supply the folder holding unet/ and vae/ as the local path.");
        }
        string folder = PaintFolder(spec.LocalPath);
        lock (_gate)
        {
            if (_paintPipelines.TryGetValue(folder, out HunyuanPaintLoader? cached))
            {
                return cached;
            }
            try
            {
                string? dino = FindPaintDino(folder);
                bool upscale = EngineKnobs.PaintUpscaleViews.Value;
                string? upscaler = PaintUpscaler(folder, upscale);
                if (upscale && upscaler is null)
                {
                    Logs.Warning($"numerics.paintUpscaleViews is on but {Path.Combine(folder, "RealESRGAN_x4plus.pth")} is missing (it is lllyasviel/Annotators RealESRGAN_x4plus.pth, not downloaded with the model): the views are baked without the upscaler.");
                }
                HunyuanPaintLoader loader = HunyuanPaintLoader.Load(_engine.Backend, folder, dino, upscaler);
                _paintPipelines[folder] = loader;
                return loader;
            }
            catch (Exception ex)
            {
                Logs.Error($"Failed to prepare the '{id}' 3D pipeline from {spec.LocalPath}", ex);
                throw;
            }
        }
    }

    /// <summary>The Real-ESRGAN checkpoint the Paint 2.1 views are upscaled with: only when the <c>numerics.paintUpscaleViews</c> knob is on and <c>RealESRGAN_x4plus.pth</c> is in
    /// <paramref name="folder"/>; otherwise <c>null</c> and the views are baked as generated.</summary>
    internal static string? PaintUpscaler(string folder, bool enabled)
    {
        string path = Path.Combine(folder, "RealESRGAN_x4plus.pth");
        return enabled && File.Exists(path) ? path : null;
    }

    // DINOv2-giant for Paint 2.1: the standalone dino/model.safetensors, else the F16 copy inside an installed Hunyuan3D shape checkpoint (same tensor names).
    internal static string? FindPaintDino(string folder)
    {
        string own = Path.Combine(folder, "dino", "model.safetensors");
        if (File.Exists(own))
        {
            return own;
        }
        string? models3D = Path.GetDirectoryName(folder.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        if (models3D is null)
        {
            return null;
        }
        foreach (string relative in new[] { "hunyuan3d/model.fp16.safetensors", "hunyuan3d2/hunyuan3d-dit-v2-0/model.fp16.safetensors", "hunyuan3d-fast/model.fp16.safetensors", "hunyuan3d-turbo/model.fp16.safetensors" })
        {
            string candidate = Path.Combine(models3D, relative);
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }
        return null;
    }

    // The folder holding unet/ and vae/: the path itself, or the nearest parent of a file (any asset of the model may be the representative one).
    private static string PaintFolder(string localPath)
    {
        if (!File.Exists(localPath))
        {
            return localPath;
        }
        string? directory = Path.GetDirectoryName(Path.GetFullPath(localPath));
        while (directory is not null && !Directory.Exists(Path.Combine(directory, "unet")))
        {
            directory = Path.GetDirectoryName(directory);
        }
        return directory ?? throw new DirectoryNotFoundException($"No unet/ folder above '{localPath}'.");
    }

    /// <summary>TRELLIS.2 image to mesh: <see cref="MeshRequest.GridResolution"/> is the final dual-grid resolution (512, 1024 or 1536; 0 is
    /// 1024), checked with the image, seed and texture size before the checkpoints load. A shape-only request returns the mesh; with
    /// <see cref="MeshRequest.Texture"/> it returns a textured PBR GLB.</summary>
    private Task<MeshResult> GenerateTrellis2Async(ModelSpec spec, MeshRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        ImageData source = request.Image!;
        Trellis2InputImage image = new() { Rgb = source.Rgb, Alpha = source.Alpha, Width = source.Width, Height = source.Height };
        Trellis2Options options = Trellis2RequestMapper.ToOptions(request);
        string modelsRoot = RepoPaths.ModelsRoot();

        return Task.Run(
            () =>
            {
                cancel.ThrowIfCancellationRequested();
                Trellis2Runtime runtime = GetOrLoadTrellis2(spec, modelsRoot);
                // One run at a time per loaded model; a queued request gives up when its own cancellation fires.
                runtime.Gate.Wait(cancel);
                try
                {
                    if (runtime.IsDisposed)
                        throw new ModelUnloadedException("The TRELLIS.2 model was unloaded while this request was queued; retry the request.");
                    Action<GenerationProgress> report = p =>
                    {
                        cancel.ThrowIfCancellationRequested();
                        progress?.Report(MeshProgress.Map(MeshModelIds.Trellis2, options.Texture, options.Grid, p));
                    };
                    if (options.Texture)
                    {
                        PbrPostProcessResult textured = runtime.Pipeline.GenerateTextured(image, options, report, cancel);
                        return new MeshResult { Data = Trellis2ImageTo3DPipeline.ToGlb(textured), Format = "glb" };
                    }

                    Mesh mesh = runtime.Pipeline.Generate(image, options, report);
                    return new MeshResult { Data = GlbWriter.Write(mesh), Format = "glb" };
                }
                finally
                {
                    runtime.Gate.Release();
                }
            },
            cancel);
    }

    /// <summary>Pixal3D image to mesh: <see cref="MeshRequest.GridResolution"/> is the final dual-grid resolution (1024 or 1536; 0 is 1024; 512
    /// has no Pixal3D pipeline), <see cref="MeshRequest.FieldOfViewRadians"/> the camera; with <see cref="MeshRequest.Texture"/> it returns a textured PBR
    /// GLB; all checked with the image and seed before the checkpoints load.</summary>
    private Task<MeshResult> GeneratePixal3DAsync(ModelSpec spec, MeshRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        ImageData source = request.Image!;
        Pixal3DOptions options = Pixal3DRequestMapper.ToOptions(request);
        Trellis2InputImage image = new() { Rgb = source.Rgb, Alpha = source.Alpha, Width = source.Width, Height = source.Height };
        string modelsRoot = RepoPaths.ModelsRoot();

        return Task.Run(
            () =>
            {
                cancel.ThrowIfCancellationRequested();
                Pixal3DRuntime runtime = GetOrLoadPixal3D(spec, modelsRoot);
                void Report(GenerationProgress p)
                {
                    cancel.ThrowIfCancellationRequested();
                    progress?.Report(MeshProgress.Map(MeshModelIds.Pixal3D, options.Texture, options.Grid, p));
                }

                return RunGated(
                    runtime,
                    "Pixal3D",
                    cancel,
                    pipeline => options.Texture
                        ? Pixal3DImageTo3DPipeline.ToGlb(pipeline.GenerateTextured(image, options, Report, cancel))
                        : GlbWriter.Write(pipeline.Generate(image, options, Report)));
            },
            cancel);
    }

    /// <summary>
    /// SAM 3D Objects: the image's alpha plane is the mask of the object to lift, MoGe estimates the pointmap, and the result is a Gaussian-splat PLY in the object's own frame (the pose is
    /// logged, not stored in the file). The pipeline opens each checkpoint for its stage and releases it, so nothing stays loaded between requests; requests run one at a time.
    /// </summary>
    private Task<MeshResult> GenerateSam3dAsync(ModelSpec spec, MeshRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        string directory = spec.LocalPath ?? throw new FileNotFoundException("No checkpoint folder found for 'sam3d-objects'. Download the catalog assets or supply the model directory as the local path.");
        if (!Sam3dObjectsPipeline.HasWeights(directory))
            throw new FileNotFoundException($"'{directory}' lacks SAM 3D checkpoints: ss_generator, ss_decoder, slat_generator and slat_decoder_gs (.safetensors or .ckpt) must all be there.");
        if (!Sam3dObjectsPipeline.HasMoge(directory))
            throw new FileNotFoundException($"'{directory}' lacks moge_vitl.safetensors, which estimates the scene's pointmap.");
        if (request.Mesh && !Sam3dObjectsPipeline.HasMeshDecoder(directory))
            throw new FileNotFoundException($"'{directory}' lacks slat_decoder_mesh.safetensors, which the mesh output needs.");
        Sam3dObjectsRequest job = Sam3dObjectsRequestMapper.ToRequest(request);
        int total = job.StructureSteps + job.SlatSteps;
        // The service outlives Dispose (the engine calls it to free memory and on a backend switch and keeps the instance), so a request is rejected only when a disposal happened after it was queued.
        int disposalsWhenQueued = Volatile.Read(ref _disposals);

        return Task.Run(
            async () =>
            {
                await _sam3dGate.WaitAsync(cancel).ConfigureAwait(false);
                try
                {
                    cancel.ThrowIfCancellationRequested();
                    if (Volatile.Read(ref _disposals) != disposalsWhenQueued)
                        throw new ModelUnloadedException("The service was disposed while this SAM 3D request was queued; retry the request.");
                    using Sam3dObjectsPipeline pipeline = new(_engine.Backend, directory);
                    Sam3dObjectsResult result = pipeline.Generate(
                        job,
                        line => Logs.Info($"[sam3d-objects] {line}"),
                        (stage, step) =>
                        {
                            cancel.ThrowIfCancellationRequested();
                            progress?.Report(MeshProgress.Plain(stage == "structure" ? step : job.StructureSteps + step, total, stage == "structure" ? "Structure flow" : "SLAT flow", step, stage == "structure" ? job.StructureSteps : job.SlatSteps, stage == "structure" ? 1 : 2, 2));
                        });
                    if (result.Mesh is { FaceCount: > 0 } meshResult)
                    {
                        Logs.Info($"[sam3d-objects] pose t=({result.Pose.TranslationX:F3}, {result.Pose.TranslationY:F3}, {result.Pose.TranslationZ:F3}) scale {result.Pose.ScaleX:F3}, mesh {meshResult.VertexCount} vertices and {meshResult.FaceCount} triangles from {result.Voxels} voxels");
                        return new MeshResult { Data = GlbWriter.Write(Sam3dObjectsMesh.ToMesh(meshResult), vertexColors: true), Format = "glb" };
                    }

                    if (request.Mesh)
                        throw new InvalidOperationException("SAM 3D's mesh decoder found no surface for this mask (try splats, or a larger or cleaner mask).");
                    if (result.Splats is not { Count: > 0 } splats)
                        throw new InvalidOperationException("SAM 3D found no occupied voxel for this mask (try a larger or cleaner mask).");
                    Logs.Info($"[sam3d-objects] pose t=({result.Pose.TranslationX:F3}, {result.Pose.TranslationY:F3}, {result.Pose.TranslationZ:F3}) scale {result.Pose.ScaleX:F3}, {splats.Count} splats from {result.Voxels} voxels");
                    return new MeshResult { Data = PlyWriter.WriteSplats(splats), Format = "ply" };
                }
                finally
                {
                    _sam3dGate.Release();
                }
            },
            cancel);
    }

    /// <summary>Runs one generation on <paramref name="runtime"/>, one at a time per loaded model; a queued request gives up when its own cancellation fires.</summary>
    private static MeshResult RunGated<TPipeline>(
        MatterGatedRuntime<TPipeline> runtime, string label, CancellationToken cancel, Func<TPipeline, byte[]> generate) where TPipeline : IDisposable
    {
        runtime.Gate.Wait(cancel);
        try
        {
            if (runtime.IsDisposed)
                throw new ModelUnloadedException($"The {label} model was unloaded while this request was queued; retry the request.");
            return new MeshResult { Data = generate(runtime.Pipeline), Format = "glb" };
        }
        finally
        {
            runtime.Gate.Release();
        }
    }

    /// <summary>Loads (or returns the cached) Pixal3D pipeline for <paramref name="spec"/>.</summary>
    private Pixal3DRuntime GetOrLoadPixal3D(ModelSpec spec, string modelsRoot)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                "No checkpoint found for 'pixal3d'. Download the catalog assets (and place naf_release.pth beside them) or supply the model directory as the local path.");
        }
        lock (_gate)
        {
            if (_pixal3d.TryGetValue(spec.LocalPath, out Pixal3DRuntime? cached))
                return cached;
            try
            {
                Pixal3DRuntime runtime = Pixal3DRuntime.Load(_engine.Backend, spec.LocalPath, modelsRoot);
                _pixal3d[spec.LocalPath] = runtime;
                return runtime;
            }
            catch (Exception ex)
            {
                Logs.Error($"Failed to load the 'pixal3d' 3D pipeline from {spec.LocalPath}", ex);
                throw;
            }
        }
    }

    /// <summary>Loads (or returns the cached) TRELLIS.2 pipeline for <paramref name="spec"/>.</summary>
    private Trellis2Runtime GetOrLoadTrellis2(ModelSpec spec, string modelsRoot)
    {
        if (spec.LocalPath is null)
        {
            throw new FileNotFoundException(
                "No checkpoint found for 'trellis2'. Download the catalog assets or supply the model directory as the local path.");
        }
        lock (_gate)
        {
            if (_trellis2.TryGetValue(spec.LocalPath, out Trellis2Runtime? cached))
                return cached;
            try
            {
                Trellis2Runtime runtime = Trellis2Runtime.Load(_engine.Backend, spec.LocalPath, modelsRoot);
                _trellis2[spec.LocalPath] = runtime;
                return runtime;
            }
            catch (Exception ex)
            {
                Logs.Error($"Failed to load the 'trellis2' 3D pipeline from {spec.LocalPath}", ex);
                throw;
            }
        }
    }

    /// <summary>The lower-case catalog id a spec names.</summary>
    internal static string IdOf(ModelSpec spec) => (spec.Catalog?.Id ?? spec.Requested ?? string.Empty).ToLowerInvariant();

    /// <summary>Whether the BiRefNet matting checkpoint that mattes images without alpha is installed right now.</summary>
    internal static bool MatterAvailable() =>
        Trellis2Runtime.MatterAvailable(RepoPaths.ModelsRoot());

    private static bool IsHairDeck(ModelSpec spec) =>
        IdOf(spec) == MeshModelIds.HairDeck;

    /// <summary>One HairDeck request: the preprocessing of the image, then the image model; the GLB carries the chains when they were made.</summary>
    private byte[] GenerateHairDeck(ModelSpec spec, MeshRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
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
            return RunHairDeck(runtime, options, request, progress, cancel);
        }
        finally
        {
            runtime.Gate.Release();
        }
    }

    private byte[] RunHairDeck(HairDeckRuntime runtime, HairDeckOptions options, MeshRequest request, IProgress<StepPreview>? progress, CancellationToken cancel)
    {
        HairDeckSourceImage source = HairDeckSourceImage.Create(request.Image!.Rgb, request.Image.Alpha, request.Image.Width, request.Image.Height);
        // A request cancelled while it queued for the gate must not start SAM 3 and BiRefNet.
        cancel.ThrowIfCancellationRequested();
        int chainSteps = options.Chains ? HairDeckCore.ChainSteps : 0;
        int total = 2 + options.Steps + chainSteps;
        int stages = options.Chains ? 4 : 3;
        HairDeckPreprocessResult prep = runtime.Preprocessor.Process(source, cancel, name =>
            progress?.Report(MeshProgress.Plain(name == HairDeckPreprocessor.SegmentStage ? 0 : 1, total, name, 0, 1, name == HairDeckPreprocessor.SegmentStage ? 1 : 2, stages)));
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
        HairDeckResult result = runtime.Model.Generate(prep.Input, options, cancel, (stage, done, count) =>
            progress?.Report(MeshProgress.Plain(
                2 + (stage == HairDeckCore.CardStage ? 0 : options.Steps) + done, total, stage, done, count, stage == HairDeckCore.CardStage ? 3 : 4, stages)));
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
        string visionRoot = ModelFolders.Under(RepoPaths.ModelsRoot(), ModelFolders.Vision);
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
                    $"3D model '{id}' has no wired image pipeline; supported: {string.Join(", ", MeshModelIds.Described)}.");
            }
            try
            {
                LoadedMesh loaded;
                if (MeshModelIds.Hunyuan3D.Contains(id))
                {
                    Hunyuan3DShapePipeline hunyuan = Hunyuan3DShapePipeline.LoadFromPath(
                        _engine.Backend, spec.LocalPath, flashVdmVaeDir: FlashVdmVaeDirectory(spec));
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

    /// <summary>The turbo VAE folder of a build whose catalog assets carry one (Turbo decodes with FlashVDM), else null.</summary>
    private static string? FlashVdmVaeDirectory(ModelSpec spec)
    {
        ModelAsset? vae = spec.Catalog?.Assets?.FirstOrDefault(a => a.Role == Registry.ModelCatalog.Hunyuan3DFlashVdmVaeRole);
        if (vae is null)
            return null;
        string? located = ModelDownloader.FindExisting(vae, ignoreCase: false) ?? ModelDownloader.FindExisting(vae, ignoreCase: true);
        return located is null
            ? throw new FileNotFoundException($"'{spec.Catalog!.Id}' decodes with FlashVDM and needs its turbo VAE; fetch it with `hartsy pull {spec.Catalog.Id}`.")
            : Path.GetDirectoryName(located);
    }

    /// <summary>Converts the wire prompt to the pipeline's prompt, checking that exactly one kind is given.</summary>
    internal static CoLa3DPrompt ToCoLa3DPrompt(SceneMeshPrompt prompt)
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
        // A SAM 3D run builds its pipeline on the engine's backend: count the disposal (a request queued before it is then rejected, one queued after it runs) and wait for the run in
        // progress, which can take minutes, as a HairDeck disposal does.
        Interlocked.Increment(ref _disposals);
        _sam3dGate.Wait();
        _sam3dGate.Release();

        // Take the entries out under the lock, then dispose outside it: a HairDeck disposal waits for a run in progress,
        // and holding the lock meanwhile would stall every other model load.
        List<LoadedMesh> pipelines;
        List<CoLa3DPipeline> scenePipelines;
        List<UniRigPipeline> rigPipelines;
        List<HunyuanPaintLoader> paintPipelines;
        List<P3SamPipeline> partPipelines;
        List<HairDeckRuntime> hairDecks;
        List<Trellis2Runtime> trellis2Runtimes;
        List<Pixal3DRuntime> pixal3dRuntimes;
        lock (_gate)
        {
            pipelines = _pipelines.Values.ToList();
            scenePipelines = _scenePipelines.Values.ToList();
            rigPipelines = _rigPipelines.Values.ToList();
            paintPipelines = _paintPipelines.Values.ToList();
            partPipelines = _partPipelines.Values.ToList();
            _xpartPipelines.Clear(); // XPartPipeline holds no resources: each run loads and releases its checkpoints
            hairDecks = _hairDecks.Values.ToList();
            trellis2Runtimes = _trellis2.Values.ToList();
            pixal3dRuntimes = _pixal3d.Values.ToList();
            _pipelines.Clear();
            _scenePipelines.Clear();
            _rigPipelines.Clear();
            _paintPipelines.Clear();
            _partPipelines.Clear();
            _hairDecks.Clear();
            _trellis2.Clear();
            _pixal3d.Clear();
        }
        // The pipelines without a run gate wait for the counted runs; the gated runtimes below wait on their own gate.
        WhenNoCountedRun(() =>
        {
            foreach (LoadedMesh loaded in pipelines)
                loaded.Pipeline.Dispose();
            foreach (CoLa3DPipeline pipeline in scenePipelines)
                pipeline.Dispose();
            foreach (UniRigPipeline pipeline in rigPipelines)
                pipeline.Dispose();
            foreach (HunyuanPaintLoader loader in paintPipelines)
                loader.Dispose();
            foreach (P3SamPipeline pipeline in partPipelines)
                pipeline.Dispose();
        });
        foreach (HairDeckRuntime runtime in hairDecks)
        {
            runtime.Dispose();
        }
        foreach (Trellis2Runtime runtime in trellis2Runtimes)
        {
            runtime.Dispose();
        }
        foreach (Pixal3DRuntime runtime in pixal3dRuntimes)
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
