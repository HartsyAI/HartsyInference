using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>The per-mode input rules for <see cref="VisionRequest"/> and <see cref="VisionBatchRequest"/>, shared by the
/// service, the CLI and the API so each refuses the same requests with the same message. Model-level limits, such as
/// whether a depth model accepts several views, are checked after this by the model itself.</summary>
/// <remarks>
/// <list type="table">
/// <listheader><term>Mode</term><description>Accepts</description></listheader>
/// <item><term>Embed … Upscale, MetricDepth, Geometry, Pose, Matting, Face, Intrinsic</term><description><see cref="VisionRequest"/>. Target size only for Upscale.</description></item>
/// <item><term>Depth, MetricDepth, Geometry</term><description>also <see cref="VisionBatchRequest"/> with <see cref="VisionSequenceKind.Views"/>, 2 to <see cref="MaxViews"/> views, no times, prompts or interpolation fields.</description></item>
/// <item><term>Track</term><description><see cref="VisionBatchRequest"/> with <see cref="VisionSequenceKind.Frames"/>, timestamps or fps, at least one non-empty in-bounds prompt; no interpolation fields.</description></item>
/// <item><term>Interpolate</term><description><see cref="VisionBatchRequest"/> with <see cref="VisionSequenceKind.Frames"/>, at least 2 frames, timestamps or fps, exactly one of factor or target fps; no prompts.</description></item>
/// </list>
/// </remarks>
public static class VisionRequestValidator
{
    /// <summary>The most frames a sequence request may carry.</summary>
    public const int MaxFrames = 4096;

    /// <summary>The most views a multi-view request may carry.</summary>
    public const int MaxViews = 64;

    /// <summary>The longest image side accepted.</summary>
    public const int MaxSide = 8192;

    /// <summary>The largest interpolation factor accepted.</summary>
    public const int MaxInterpolationFactor = 16;

    private static readonly HashSet<VisionMode> SequenceOnlyModes = [VisionMode.Track, VisionMode.Interpolate];

    private static readonly HashSet<VisionMode> MultiViewModes = [VisionMode.Depth, VisionMode.MetricDepth, VisionMode.Geometry];

    /// <summary>True when <paramref name="mode"/> only accepts a frame sequence.</summary>
    public static bool IsSequenceOnly(VisionMode mode) => SequenceOnlyModes.Contains(mode);

    /// <summary>True when <paramref name="mode"/> may take several views of one scene.</summary>
    public static bool AcceptsViews(VisionMode mode) => MultiViewModes.Contains(mode);

    /// <summary>Throws <see cref="ArgumentException"/> naming the first rule <paramref name="request"/> breaks.</summary>
    public static void Validate(VisionRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (!Enum.IsDefined(request.Mode))
            throw new ArgumentException($"Unknown vision mode {(int)request.Mode}.");
        if (IsSequenceOnly(request.Mode))
        {
            throw new ArgumentException(
                $"{request.Mode} needs a frame sequence: send a VisionBatchRequest with Kind = Frames, not a single image.");
        }
        CheckImage(request.Image, "Image");
        CheckThreshold(request.Threshold);
        if (request.Mode != VisionMode.Upscale && (request.TargetWidth is not null || request.TargetHeight is not null))
            throw new ArgumentException($"A target size only applies to Upscale, not {request.Mode}.");
        if (request.TargetWidth is <= 0 || request.TargetHeight is <= 0)
            throw new ArgumentException("A target width or height must be positive.");
    }

    /// <summary>Throws <see cref="ArgumentException"/> naming the first rule <paramref name="request"/> breaks.</summary>
    public static void Validate(VisionBatchRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (!Enum.IsDefined(request.Mode))
            throw new ArgumentException($"Unknown vision mode {(int)request.Mode}.");
        if (request.Images is not { Count: > 0 } images)
            throw new ArgumentException("A batch request needs at least one image.");
        for (int i = 0; i < images.Count; i++)
            CheckImage(images[i], $"Images[{i}]");
        int width = images[0].Width, height = images[0].Height;
        for (int i = 1; i < images.Count; i++)
        {
            if (images[i].Width != width || images[i].Height != height)
            {
                throw new ArgumentException(
                    $"Every image must share one size: Images[0] is {width}x{height}, Images[{i}] is {images[i].Width}x{images[i].Height}.");
            }
        }
        CheckThreshold(request.Threshold);
        bool interpolationSet = request.InterpolationFactor is not null || request.TargetFps is not null;
        if (request.Mode != VisionMode.Interpolate && interpolationSet)
            throw new ArgumentException($"An interpolation factor or target fps only applies to Interpolate, not {request.Mode}.");

        if (request.Kind == VisionSequenceKind.Views)
        {
            ValidateViews(request, images.Count);
            return;
        }
        if (request.Kind != VisionSequenceKind.Frames)
            throw new ArgumentException($"Unknown sequence kind {(int)request.Kind}.");
        if (!IsSequenceOnly(request.Mode))
        {
            throw new ArgumentException(
                $"{request.Mode} does not take a frame sequence: send each frame as a VisionRequest. Frames are for Track and Interpolate.");
        }
        if (images.Count > MaxFrames)
            throw new ArgumentException($"{images.Count} frames is over the limit of {MaxFrames}.");
        ValidateTiming(request, images.Count);
        if (request.Mode == VisionMode.Interpolate)
            ValidateInterpolation(request, images.Count);
        else
            ValidatePrompts(request, images.Count, width, height);
    }

    private static void ValidateViews(VisionBatchRequest request, int count)
    {
        if (!AcceptsViews(request.Mode))
            throw new ArgumentException($"{request.Mode} does not take several views. Multi-view is for Depth, MetricDepth and Geometry.");
        if (count < 2)
            throw new ArgumentException("A multi-view request needs at least 2 views; send one view as a VisionRequest.");
        if (count > MaxViews)
            throw new ArgumentException($"{count} views is over the limit of {MaxViews}.");
        if (request.TimestampsSeconds is not null || request.Fps is not null)
            throw new ArgumentException("Views have no time order: timestamps and fps only apply to frames.");
        if (request.Prompts is { Count: > 0 })
            throw new ArgumentException("Prompts only apply to Track frames, not views.");
    }

    private static void ValidateTiming(VisionBatchRequest request, int count)
    {
        if ((request.TimestampsSeconds is null) == (request.Fps is null))
            throw new ArgumentException("Frames need exactly one of TimestampsSeconds and Fps.");
        if (request.Fps is { } fps && (!double.IsFinite(fps) || fps <= 0))
            throw new ArgumentException($"Fps must be a positive number; got {fps}.");
        if (request.TimestampsSeconds is not { } times)
            return;
        if (times.Count != count)
            throw new ArgumentException($"{times.Count} timestamps for {count} frames: give one per frame.");
        for (int i = 0; i < times.Count; i++)
        {
            if (!double.IsFinite(times[i]))
                throw new ArgumentException($"TimestampsSeconds[{i}] is not a finite number.");
            if (i > 0 && times[i] <= times[i - 1])
                throw new ArgumentException($"Timestamps must strictly increase: [{i - 1}] is {times[i - 1]}, [{i}] is {times[i]}.");
        }
    }

    private static void ValidateInterpolation(VisionBatchRequest request, int count)
    {
        if (count < 2)
            throw new ArgumentException("Interpolate needs at least 2 frames.");
        if (request.Prompts is { Count: > 0 })
            throw new ArgumentException("Interpolate takes no prompts.");
        if ((request.InterpolationFactor is null) == (request.TargetFps is null))
            throw new ArgumentException("Interpolate needs exactly one of InterpolationFactor and TargetFps.");
        if (request.InterpolationFactor is { } factor && (factor < 2 || factor > MaxInterpolationFactor))
            throw new ArgumentException($"InterpolationFactor must be 2 to {MaxInterpolationFactor}; got {factor}.");
        if (request.TargetFps is { } target)
        {
            if (!double.IsFinite(target) || target <= 0)
                throw new ArgumentException($"TargetFps must be a positive number; got {target}.");
            double sourceFps = request.Fps ?? (count - 1) / (request.TimestampsSeconds![^1] - request.TimestampsSeconds[0]);
            if (target <= sourceFps)
                throw new ArgumentException($"TargetFps {target} must be above the source rate ({sourceFps:0.###} fps).");
        }
    }

    private static void ValidatePrompts(VisionBatchRequest request, int count, int width, int height)
    {
        if (request.Prompts is not { Count: > 0 } prompts || prompts.All(p => p.IsEmpty))
            throw new ArgumentException("Track needs at least one prompt with text, points or boxes.");
        foreach (VisionFramePrompt prompt in prompts)
        {
            if (prompt.FrameIndex < 0 || prompt.FrameIndex >= count)
                throw new ArgumentException($"A prompt names frame {prompt.FrameIndex}, but there are {count} frames.");
            foreach (VisionPoint p in prompt.Points ?? [])
            {
                if (!float.IsFinite(p.X) || !float.IsFinite(p.Y) || p.X < 0 || p.Y < 0 || p.X > width || p.Y > height)
                    throw new ArgumentException($"Point ({p.X}, {p.Y}) on frame {prompt.FrameIndex} is outside the {width}x{height} frame.");
            }
            foreach (VisionBox b in prompt.Boxes ?? [])
            {
                if (!(b.X0 < b.X1 && b.Y0 < b.Y1) || b.X0 < 0 || b.Y0 < 0 || b.X1 > width || b.Y1 > height)
                    throw new ArgumentException($"Box ({b.X0}, {b.Y0})-({b.X1}, {b.Y1}) on frame {prompt.FrameIndex} is empty or outside the {width}x{height} frame.");
            }
        }
    }

    private static void CheckImage(ImageData? image, string name)
    {
        if (image is null)
            throw new ArgumentException($"{name} is missing.");
        if (image.Width <= 0 || image.Height <= 0 || image.Width > MaxSide || image.Height > MaxSide)
            throw new ArgumentException($"{name} is {image.Width}x{image.Height}; each side must be 1 to {MaxSide} pixels.");
        if (image.Rgb is null || image.Rgb.Length != image.Width * image.Height * 3)
            throw new ArgumentException($"{name} must carry {image.Width}x{image.Height}x3 RGB bytes.");
    }

    private static void CheckThreshold(double threshold)
    {
        if (!double.IsFinite(threshold) || threshold < 0 || threshold > 1)
            throw new ArgumentException($"Threshold must be in [0, 1]; got {threshold}.");
    }
}
