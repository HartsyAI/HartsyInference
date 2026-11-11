using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Worlds;

/// <summary>The one planning step for world requests. It checks a request against the model's declared
/// <see cref="WorldCapabilities"/> and resolves its defaults, without touching a weight. The CLI and the HTTP route
/// both call it, and <see cref="Services.IWorldService.Open"/> calls it again on the plan it is handed, which is
/// idempotent on an already resolved request.</summary>
public static class WorldPlanResolver
{
    /// <summary>Resolves <paramref name="request"/> for <paramref name="spec"/> on the backend <paramref name="backendSelector"/> names.</summary>
    /// <param name="spec">The model; its catalog entry supplies the capabilities, and a catalog-less spec gets the defaults.</param>
    /// <param name="request">The caller's request, unmodified.</param>
    /// <param name="backendSelector">A backend selector (<c>cpu</c>, <c>cuda:1</c>, <c>auto</c>); <c>auto</c> is resolved to a concrete kind.</param>
    /// <param name="drawSeed">Source of a seed when the request names none; null draws one at random.</param>
    /// <exception cref="WorldPlanningException">The request asks for something the model does not accept, or a value
    /// outside its declared set.</exception>
    public static WorldPlan Resolve(ModelSpec spec, WorldRequest request, string backendSelector, Func<long>? drawSeed = null)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(backendSelector);
        ImageData init = request.InitImage
            ?? throw new WorldPlanningException("World models roll out from a first frame; set WorldRequest.InitImage.");
        string id = spec.Catalog?.Id ?? spec.Requested;
        WorldCapabilities caps = spec.Catalog?.World ?? new WorldCapabilities();
        List<string> notes = [];

        string backend = ResolveBackend(caps, backendSelector, id);
        string precision = request.Precision is null ? caps.Precisions[0] : Allowed(caps.Precisions, request.Precision, "precision", id);
        RejectUnsupportedOptions(caps, request, id);
        bool streaming = ResolveStreaming(caps, request, id);
        int requestedFrames = request.Unbounded ? 0 : request.Frames ?? caps.DefaultFrames;
        int frames = request.Unbounded ? 0 : ResolveFrames(caps, requestedFrames, id, notes);
        int fps = ResolveFps(caps, request.Fps ?? caps.DefaultFps, id);
        int steps = ResolveSteps(caps, request.Steps, id, notes);
        string? sampler = request.Sampler is null ? null : Allowed(caps.Samplers, request.Sampler, "sampler", id);
        // A stream steered by live actions has no path to check up front: an empty path or a script is its opening state.
        bool live = streaming && caps.Streaming!.SupportsLiveActions && request.Camera is not { Poses.Count: > 0 };
        if (request.Unbounded && !live)
        {
            throw new WorldPlanningException(
                $"An unbounded '{id}' stream is steered by live actions; it takes no fixed camera path. Give a frame count with the path.");
        }
        CameraPlan camera = live
            ? ResolveLiveCamera(caps, request.Camera, init, id)
            : ResolveCamera(caps, request.Camera, frames, init, id, notes);

        bool seedDrawn = request.Seed < 0;
        long seed = seedDrawn ? (drawSeed?.Invoke() ?? Random.Shared.NextInt64(0, int.MaxValue)) : request.Seed;
        WorldRequest resolved = request with
        {
            Frames = request.Unbounded ? null : frames,
            Streaming = streaming,
            Fps = fps,
            Steps = steps,
            Seed = seed,
            Precision = precision,
            Sampler = sampler,
            Camera = camera.Path,
        };
        return new WorldPlan
        {
            ModelId = id,
            Backend = backend,
            Precision = precision,
            RequestedFrames = requestedFrames,
            Frames = frames,
            Streaming = streaming,
            Unbounded = request.Unbounded,
            Fps = fps,
            Steps = steps,
            Guidance = request.Guidance,
            FlowShift = request.FlowShift,
            Sampler = sampler,
            Refiner = request.Refiner,
            ProgressivePreview = request.ProgressivePreview,
            SequentialCfg = request.SequentialCfg,
            Seed = seed,
            SeedDrawn = seedDrawn,
            CameraPoses = camera.Path?.Poses.Count ?? 0,
            CameraTruncatedFrom = camera.TruncatedFrom,
            Intrinsics = camera.Path?.Intrinsics ?? camera.Path?.FrameIntrinsics?[0],
            FrameIntrinsics = camera.Path?.FrameIntrinsics?.Count ?? 0,
            Notes = notes,
            Request = resolved,
        };
    }

    private static string ResolveBackend(WorldCapabilities caps, string selector, string id)
    {
        string kind = BackendFactory.Kind(selector);
        if (kind == "auto")
        {
            kind = BackendFactory.Resolve(selector);
        }
        if (caps.Backends is { Count: > 0 } allowed && !allowed.Contains(kind, StringComparer.OrdinalIgnoreCase))
        {
            throw new WorldPlanningException($"'{id}' does not run on the {kind} backend; allowed: {string.Join(", ", allowed)}.");
        }
        return kind;
    }

    /// <summary>Whether the session streams: the request's choice, else the model's only mode. A streaming request needs a model
    /// that declares streaming, a batch request a model that has a batch mode, and an unbounded run a stream that supports it.</summary>
    private static bool ResolveStreaming(WorldCapabilities caps, WorldRequest request, string id)
    {
        WorldStreamingCapabilities? streaming = caps.Streaming;
        bool wanted = request.Streaming ?? streaming?.Only ?? false;
        if (wanted && streaming is null)
        {
            throw new WorldPlanningException(
                $"'{id}' is a batch model: it renders its whole rollout in one run and has no chunk-by-chunk streaming mode.");
        }
        if (!wanted && streaming is { Only: true })
        {
            throw new WorldPlanningException($"'{id}' only generates as a stream; it has no whole-clip batch mode.");
        }
        if (!request.Unbounded)
        {
            return wanted;
        }
        if (!wanted || streaming is not { SupportsUnbounded: true })
        {
            throw new WorldPlanningException(
                $"'{id}' needs a frame count; only a streaming model that supports unbounded runs can run until it is stopped.");
        }
        if (request.Frames is not null)
        {
            throw new WorldPlanningException("Give a frame count or an unbounded run, not both.");
        }
        return true;
    }

    /// <summary>The camera of a stream steered by live actions: no poses, an optional opening script kept as written (the
    /// stream plays it, then follows the actions it is sent), and one set of intrinsics.</summary>
    private static CameraPlan ResolveLiveCamera(WorldCapabilities caps, WorldCameraPath? camera, ImageData init, string id)
    {
        if (camera?.FrameIntrinsics is not null)
        {
            throw new WorldPlanningException($"A live '{id}' stream takes one set of intrinsics for every frame, not per-frame intrinsics.");
        }
        if (camera?.Script is string script)
        {
            if (!caps.CameraModes.Contains(WorldCameraMode.ActionScript) || caps.CameraScript is null)
            {
                throw new WorldPlanningException($"'{id}' takes no action-script camera path.");
            }
            try
            {
                _ = caps.CameraScript.Expand(script);
            }
            catch (FormatException ex)
            {
                throw new WorldPlanningException($"The camera action script for '{id}' is invalid: {ex.Message}");
            }
        }
        WorldIntrinsics? intrinsics = ValidateIntrinsics(camera?.Intrinsics, caps, init, id);
        return new CameraPlan(new WorldCameraPath { Script = camera?.Script, Intrinsics = intrinsics }, null);
    }

    private static void RejectUnsupportedOptions(WorldCapabilities caps, WorldRequest request, string id)
    {
        if (request.Guidance is not null && !caps.SupportsGuidance)
        {
            throw new WorldPlanningException($"'{id}' takes no guidance option; its guidance is fixed by the model.");
        }
        if (request.FlowShift is not null && !caps.SupportsFlowShift)
        {
            throw new WorldPlanningException($"'{id}' takes no flow shift option.");
        }
        if (request.Refiner && !caps.SupportsRefiner)
        {
            throw new WorldPlanningException($"'{id}' has no refiner stage; the refiner option is not supported.");
        }
        if (request.SequentialCfg && !caps.SupportsSequentialCfg)
        {
            throw new WorldPlanningException($"'{id}' has no sequential classifier-free guidance path; leave SequentialCfg unset.");
        }
        if (request.ProgressivePreview && !caps.SupportsProgressivePreview)
        {
            throw new WorldPlanningException(
                $"Unsupported progressive inference: '{id}' is a batch model and cannot stream partial frames. "
                + "Its frames are delivered as each rollout completes; leave ProgressivePreview unset.");
        }
    }

    /// <remarks>The minimum is checked before the grid, and the maximum before <see cref="FrameGrid.Normalize"/>, so a
    /// count above the cap is refused rather than clamped to it.</remarks>
    private static int ResolveFrames(WorldCapabilities caps, int requested, string id, List<string> notes)
    {
        if (requested < caps.MinFrames)
        {
            throw new WorldPlanningException($"'{id}' needs at least {caps.MinFrames} frames (the seed frame counts); {requested} requested.");
        }
        if (caps.Frames is not FrameGrid grid)
        {
            return requested;
        }
        if (grid.Max is int max && requested > max)
        {
            throw new WorldPlanningException($"'{id}' generates at most {max} frames per rollout; {requested} requested.");
        }
        if (grid.Normalize(requested) is not int frames)
        {
            string nearest = NearestValid(caps, grid, requested);
            throw new WorldPlanningException(
                $"'{id}' takes frame counts on the {grid} grid; {requested} is not on it. Nearest valid counts: {nearest}.");
        }
        if (frames < caps.MinFrames)
        {
            throw new WorldPlanningException($"'{id}' needs at least {caps.MinFrames} frames; {requested} normalizes to {frames}.");
        }
        if (frames != requested)
        {
            notes.Add($"frames {requested} normalized to {frames} on the {grid} grid");
        }
        return frames;
    }

    private static string NearestValid(WorldCapabilities caps, FrameGrid grid, int requested)
    {
        int lower = grid.Offset + Math.Max(0, (requested - grid.Offset) / grid.Step) * grid.Step;
        List<string> counts = [];
        for (int candidate = lower; candidate <= lower + grid.Step; candidate += grid.Step)
        {
            bool inRange = candidate >= caps.MinFrames && (grid.Max is not int max || candidate <= max);
            if (inRange && candidate != requested)
            {
                counts.Add(candidate.ToString(System.Globalization.CultureInfo.InvariantCulture));
            }
        }
        return counts.Count == 0 ? "none" : string.Join(", ", counts);
    }

    private static int ResolveFps(WorldCapabilities caps, int requested, string id)
    {
        if (requested < 1)
        {
            throw new WorldPlanningException($"fps must be a positive frame rate; {requested} requested.");
        }
        if (caps.Fps is { Count: > 0 } allowed && !allowed.Contains(requested))
        {
            throw new WorldPlanningException($"'{id}' runs at {string.Join(", ", allowed)} fps; {requested} requested.");
        }
        return requested;
    }

    private static int ResolveSteps(WorldCapabilities caps, int requested, string id, List<string> notes)
    {
        if (requested < 0)
        {
            throw new WorldPlanningException($"steps must be zero (the model default) or positive; {requested} requested.");
        }
        if (caps.DefaultSteps == 0)
        {
            if (requested > 0)
            {
                notes.Add($"steps {requested} ignored: '{id}' fixes its own step count");
            }
            return 0;
        }
        return requested > 0 ? requested : caps.DefaultSteps;
    }

    private static string Allowed(IReadOnlyList<string> allowed, string value, string option, string id)
    {
        foreach (string candidate in allowed)
        {
            if (string.Equals(candidate, value, StringComparison.OrdinalIgnoreCase))
            {
                return candidate;
            }
        }
        if (allowed.Count == 0)
        {
            throw new WorldPlanningException($"'{id}' takes no {option} option; {value} was requested.");
        }
        throw new WorldPlanningException($"'{id}' does not support {option} '{value}'; allowed: {string.Join(", ", allowed)}.");
    }

    private static CameraPlan ResolveCamera(
        WorldCapabilities caps, WorldCameraPath? camera, int frames, ImageData init, string id, List<string> notes)
    {
        if (camera is null)
        {
            if (caps.CameraRequired)
            {
                string modes = caps.CameraModes.Contains(WorldCameraMode.ActionScript)
                    ? "one camera-to-world pose per frame, or an action script"
                    : "one camera-to-world pose per frame";
                throw new WorldPlanningException($"'{id}' needs a camera path: {modes}.");
            }
            return new CameraPlan(null, null);
        }
        IReadOnlyList<float[]> poses = camera.Script is null ? camera.Poses : ExpandScript(caps, camera, id, notes);
        if (camera.Script is null && !caps.CameraModes.Contains(WorldCameraMode.PerFramePose))
        {
            throw new WorldPlanningException($"'{id}' takes no camera path; camera input is not supported by this model.");
        }
        ValidatePoses(poses, id);
        if (poses.Count < frames)
        {
            throw new WorldPlanningException(
                $"The camera path has {poses.Count} poses but {frames} frames were requested; supply one pose per frame.");
        }
        int? truncatedFrom = null;
        if (poses.Count > frames)
        {
            truncatedFrom = poses.Count;
            poses = poses.Take(frames).ToArray();
            notes.Add($"camera path truncated from {truncatedFrom} to {frames} poses");
        }
        if (camera.Intrinsics is not null && camera.FrameIntrinsics is not null)
        {
            throw new WorldPlanningException("Give intrinsics for every frame or per-frame intrinsics, not both.");
        }
        WorldIntrinsics? intrinsics = null;
        IReadOnlyList<WorldIntrinsics>? frameIntrinsics = null;
        if (camera.FrameIntrinsics is not null)
        {
            if (camera.FrameIntrinsics.Count == 0)
            {
                throw new WorldPlanningException("The per-frame intrinsics list is empty.");
            }
            foreach (WorldIntrinsics entry in camera.FrameIntrinsics)
            {
                ValidateIntrinsics(entry, caps, init, id);
            }
            frameIntrinsics = camera.FrameIntrinsics;
        }
        else
        {
            intrinsics = ValidateIntrinsics(camera.Intrinsics, caps, init, id);
        }
        WorldCameraPath resolved = new WorldCameraPath { Poses = poses, Intrinsics = intrinsics, FrameIntrinsics = frameIntrinsics };
        return new CameraPlan(resolved, truncatedFrom);
    }

    /// <summary>Expands an action-script camera path into poses. The resolved request carries the poses, so resolving it again
    /// does not expand twice.</summary>
    private static IReadOnlyList<float[]> ExpandScript(WorldCapabilities caps, WorldCameraPath camera, string id, List<string> notes)
    {
        if (!caps.CameraModes.Contains(WorldCameraMode.ActionScript) || caps.CameraScript is null)
        {
            throw new WorldPlanningException($"'{id}' takes no action-script camera path.");
        }
        if (camera.Poses.Count > 0)
        {
            throw new WorldPlanningException("A camera path carries poses or an action script, not both.");
        }
        IReadOnlyList<float[]> poses;
        try
        {
            poses = caps.CameraScript.Expand(camera.Script!);
        }
        catch (FormatException ex)
        {
            throw new WorldPlanningException($"The camera action script for '{id}' is invalid: {ex.Message}");
        }
        notes.Add($"camera path expanded from action script '{camera.Script}' to {poses.Count} poses");
        return poses;
    }

    private static void ValidatePoses(IReadOnlyList<float[]> poses, string id)
    {
        if (poses.Count == 0)
        {
            throw new WorldPlanningException($"The camera path for '{id}' has no poses.");
        }
        for (int i = 0; i < poses.Count; i++)
        {
            float[]? pose = poses[i];
            if (pose is null || pose.Length != 16)
            {
                throw new WorldPlanningException(
                    $"Camera pose {i} has {pose?.Length ?? 0} values; each pose is 16 row-major values of a 4x4 camera-to-world matrix.");
            }
            foreach (float value in pose)
            {
                if (!float.IsFinite(value))
                {
                    throw new WorldPlanningException($"Camera pose {i} contains a non-finite value.");
                }
            }
        }
    }

    private static WorldIntrinsics? ValidateIntrinsics(WorldIntrinsics? intrinsics, WorldCapabilities caps, ImageData init, string id)
    {
        if (caps.Intrinsics == WorldIntrinsicsPolicy.NotAccepted)
        {
            return intrinsics is null ? null : throw new WorldPlanningException($"'{id}' takes no intrinsics.");
        }
        if (intrinsics is null)
        {
            return caps.Intrinsics == WorldIntrinsicsPolicy.Required
                ? throw new WorldPlanningException($"'{id}' needs intrinsics for camera input: fx, fy, cx, cy in pixels of the seed image.")
                : null;
        }
        if (!double.IsFinite(intrinsics.Fx) || !double.IsFinite(intrinsics.Fy) || !double.IsFinite(intrinsics.Cx)
            || !double.IsFinite(intrinsics.Cy) || intrinsics.Fx <= 0 || intrinsics.Fy <= 0)
        {
            throw new WorldPlanningException("Intrinsics need finite, positive focal lengths fx and fy.");
        }
        if (intrinsics.Cx < 0 || intrinsics.Cx > init.Width || intrinsics.Cy < 0 || intrinsics.Cy > init.Height)
        {
            throw new WorldPlanningException(
                $"The principal point ({intrinsics.Cx}, {intrinsics.Cy}) lies outside the {init.Width}x{init.Height} seed image.");
        }
        double horizontal = FieldOfViewDegrees(intrinsics.Fx, init.Width);
        double vertical = FieldOfViewDegrees(intrinsics.Fy, init.Height);
        if (!InRange(horizontal, caps) || !InRange(vertical, caps))
        {
            throw new WorldPlanningException(
                $"'{id}' field of view must be {caps.MinFovDegrees:0}-{caps.MaxFovDegrees:0} degrees; the intrinsics give "
                + $"{horizontal:0.0} horizontal and {vertical:0.0} vertical.");
        }
        return intrinsics;
    }

    private static double FieldOfViewDegrees(double focal, int pixels) =>
        2.0 * Math.Atan(pixels / (2.0 * focal)) * 180.0 / Math.PI;

    private static bool InRange(double degrees, WorldCapabilities caps) =>
        degrees >= caps.MinFovDegrees && degrees <= caps.MaxFovDegrees;

    private sealed record CameraPlan(WorldCameraPath? Path, int? TruncatedFrom);
}
