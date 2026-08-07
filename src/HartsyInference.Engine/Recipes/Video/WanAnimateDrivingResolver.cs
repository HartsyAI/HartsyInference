using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Video.Encoding;
using HartsyInference.Vision.Detection;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Resolves Wan-Animate's two driving inputs — the pose skeleton clip (<c>[1,3,T,H,W]</c>) and the cropped
/// face clip (<c>[1,3,T−1,S,S]</c>), both in [-1, 1] — from the request. Per branch the precedence is: explicit
/// pre-rendered override clip → auto-preprocess of the driving video (YOLO11-pose skeleton render / face crop, the
/// format the checkpoint was trained on) → the raw driving clip resized (face branch center-crop-squared) → the
/// tiled <see cref="VideoRequest.InitImage"/> still. Ported from the SwarmUI extension's
/// <c>WanAnimateDrivingPreprocessor</c>.</summary>
internal static class WanAnimateDrivingResolver
{
    private static int _poseWeightsWarned;

    /// <summary>The built driving clips plus the frame count they settled on and the driving video's decoded rate.</summary>
    internal sealed record ResolvedClips(Tensor PoseClip, Tensor FaceClip, int FrameCount, int? DrivingFps);

    /// <summary>Builds both driving clips; <paramref name="requestedFrames"/> is the grid-resolved request count, which a
    /// shorter driving video shrinks per <see cref="ResolveDrivingFrames"/>. The caller owns disposal of both tensors.</summary>
    internal static ResolvedClips Resolve(IBackend backend, VideoRequest request, int width, int height,
        int requestedFrames, int temporalStep, int motionSize, CancellationToken cancel)
    {
        List<byte[]>? drivingFrames = null;
        int frameCount = requestedFrames;
        int? drivingFps = null;
        if (request.DrivingVideo is not null)
        {
            FfmpegProcessDecoder.Result decoded = DecodeClip(request.DrivingVideo, width, height, requestedFrames, cancel);
            frameCount = ResolveDrivingFrames(requestedFrames, decoded.Frames.Count, temporalStep);
            drivingFrames = FitFrames(decoded.Frames, frameCount);
            drivingFps = decoded.Fps > 0.5 ? (int)Math.Round(decoded.Fps) : null;
            Logs.Info($"[WanAnimate] driving video decoded: {decoded.Frames.Count} frame(s) @ {decoded.Fps:0.##} fps → "
                + $"{frameCount} frame(s) at {width}x{height}.");
        }

        bool needsAuto = request.DrivingAutoPreprocess && drivingFrames is not null
            && (request.DrivingPoseVideo is null || request.DrivingFaceVideo is null);
        YoloPosePipeline? pose = needsAuto ? TryCreatePosePipeline(backend) : null;
        try
        {
            Tensor poseClip = BuildPoseClip(backend, request, pose, drivingFrames, width, height, frameCount, cancel);
            try
            {
                Tensor faceClip = BuildFaceClip(backend, request, pose, drivingFrames, width, height, motionSize, frameCount, cancel);
                return new ResolvedClips(poseClip, faceClip, frameCount, drivingFps);
            }
            catch
            {
                poseClip.Dispose();
                throw;
            }
        }
        finally
        {
            pose?.Dispose();
        }
    }

    /// <summary>Frame-count rule for a decoded driving video: <c>min(requested, available)</c> snapped DOWN onto the
    /// VAE's <c>step·n + 1</c> temporal grid, floored at 5 (the face pathway downsamples 4x).</summary>
    internal static int ResolveDrivingFrames(int requestedFrames, int availableFrames, int temporalStep)
    {
        int frames = Math.Min(requestedFrames, Math.Max(1, availableFrames));
        frames = 1 + (frames - 1) / temporalStep * temporalStep;
        return Math.Max(frames, 5);
    }

    /// <summary>Truncates (longer) or repeat-pads the last frame (shorter) to exactly <paramref name="count"/> frames.</summary>
    internal static List<byte[]> FitFrames(List<byte[]> frames, int count)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frames.Count == 0)
        {
            throw new ArgumentException("The clip decoded to zero frames.", nameof(frames));
        }
        while (frames.Count < count)
        {
            frames.Add(frames[^1]);
        }
        if (frames.Count > count)
        {
            frames.RemoveRange(count, frames.Count - count);
        }
        return frames;
    }

    private static Tensor BuildPoseClip(IBackend backend, VideoRequest request, YoloPosePipeline? pose,
        List<byte[]>? drivingFrames, int width, int height, int frameCount, CancellationToken cancel)
    {
        if (request.DrivingPoseVideo is not null)
        {
            Logs.Info("[WanAnimate] pose branch: using the supplied pre-rendered pose video.");
            return DecodeToClip(request.DrivingPoseVideo, width, height, frameCount, cancel);
        }
        if (drivingFrames is not null)
        {
            return pose is not null
                ? WanAnimatePoseClipBuilder.Build(backend, pose, drivingFrames, width, height, cancel)
                : VideoRecipeUtils.PackRgbFramesToClip(drivingFrames, width, height);
        }
        ImageData still = RequireStill(request);
        return VideoRecipeUtils.TileRgbToClip(VideoRecipeUtils.ResizeRgb24(still, width, height), width, height, frameCount);
    }

    private static Tensor BuildFaceClip(IBackend backend, VideoRequest request, YoloPosePipeline? pose,
        List<byte[]>? drivingFrames, int width, int height, int motionSize, int frameCount, CancellationToken cancel)
    {
        int faceFrames = frameCount - 1;
        if (request.DrivingFaceVideo is not null)
        {
            Logs.Info("[WanAnimate] face branch: using the supplied pre-cropped face video.");
            return DecodeToClip(request.DrivingFaceVideo, motionSize, motionSize, faceFrames, cancel);
        }
        if (drivingFrames is not null)
        {
            // Pose null → center-crop-squared raw fallback; the builder handles both.
            List<byte[]> leading = drivingFrames.GetRange(0, faceFrames);
            return WanAnimateFaceClipBuilder.Build(backend, pose, leading, width, height, motionSize, cancel);
        }
        ImageData still = RequireStill(request);
        return VideoRecipeUtils.TileRgbToClip(
            VideoRecipeUtils.ResizeRgb24(still, motionSize, motionSize), motionSize, motionSize, faceFrames);
    }

    private static ImageData RequireStill(VideoRequest request) =>
        request.InitImage ?? throw new InvalidOperationException(
            "Wan-Animate needs a driving motion input: set VideoRequest.DrivingVideo (a driving video) "
            + "or VideoRequest.InitImage (a still tiled across frames).");

    /// <summary>Decodes an override clip at the exact target geometry and packs it, truncated/repeat-padded to
    /// <paramref name="numFrames"/> (the extension's <c>DecodeControlClip</c> semantics).</summary>
    private static Tensor DecodeToClip(VideoClip clip, int width, int height, int numFrames, CancellationToken cancel)
    {
        FfmpegProcessDecoder.Result decoded = DecodeClip(clip, width, height, numFrames, cancel);
        return VideoRecipeUtils.PackRgbFramesToClip(FitFrames(decoded.Frames, numFrames), width, height);
    }

    private static FfmpegProcessDecoder.Result DecodeClip(VideoClip clip, int width, int height, int maxFrames, CancellationToken cancel)
    {
        try
        {
            return new FfmpegProcessDecoder()
                .DecodeAsync(clip.Data, clip.Format, maxFrames, width, height, cancel).GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            Logs.Error($"[WanAnimate] Failed to decode a driving clip (format hint '{clip.Format ?? "none"}').", ex);
            throw;
        }
    }

    /// <summary>Loads the shared YOLO11-pose pipeline for auto-preprocess, or null (→ raw-clip fallback, warned once)
    /// when the folded weights are not installed or fail to load — auto is the default, so this never throws.</summary>
    private static YoloPosePipeline? TryCreatePosePipeline(IBackend backend)
    {
        string? path = ModelFileLocator.Find(IpAdapterResolver.PoseWeightsFile, IpAdapterResolver.DetectorFolders);
        if (path is null)
        {
            if (Interlocked.Exchange(ref _poseWeightsWarned, 1) == 0)
            {
                Logs.Warning($"[WanAnimate] Auto-preprocess needs the folded YOLO11n-pose weights ('{IpAdapterResolver.PoseWeightsFile}') "
                    + "under the models root; falling back to the raw driving clip. Convert Ultralytics 'yolo11n-pose.pt' with "
                    + "tests/python-reference/convert_yolov8_pt_to_safetensors.py.");
            }
            return null;
        }
        try
        {
            return new YoloPosePipeline(backend, YoloConfig.YoloV11nPose, path, inputSize: 640);
        }
        catch (Exception ex)
        {
            Logs.Error($"[WanAnimate] Failed to load the YOLO11-pose pipeline from '{path}'; falling back to the raw driving clip.", ex);
            return null;
        }
    }
}
