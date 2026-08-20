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
    internal sealed record ResolvedClips(Tensor PoseClip, Tensor FaceClip, int FrameCount, int? DrivingFps,
        Tensor? BackgroundClip = null, Tensor? MaskClip = null);

    /// <summary>Builds both driving clips; <paramref name="requestedFrames"/> is the grid-resolved request count, which a
    /// shorter driving video shrinks per <see cref="ResolveDrivingFrames"/> unless <paramref name="pinFrameCount"/>.
    /// <paramref name="frameOffset"/> seeks every driving input (the chunked extension's <c>video_frame_offset</c>).
    /// The caller owns disposal of both tensors.</summary>
    internal static ResolvedClips Resolve(IBackend backend, VideoRequest request, int width, int height,
        int requestedFrames, int temporalStep, int motionSize, CancellationToken cancel,
        int frameOffset = 0, bool pinFrameCount = false)
    {
        List<byte[]>? drivingFrames = null;
        int frameCount = requestedFrames;
        int? drivingFps = null;
        if (request.DrivingVideo is not null)
        {
            FfmpegProcessDecoder.Result decoded = DecodeClip(request.DrivingVideo, width, height, frameOffset + requestedFrames, cancel);
            int decodedCount = decoded.Frames.Count;
            List<byte[]> available = DropLeadingFrames(decoded.Frames, frameOffset);
            if (available.Count == 0)
            {
                throw new InvalidOperationException(
                    $"The Wan-Animate driving video ran out at frame {frameOffset} (only {decodedCount} frame(s) decoded) — "
                    + "lower AnimateTotalFrames or supply a longer driving video.");
            }
            frameCount = ResolveChunkFrames(requestedFrames, available.Count, temporalStep, pinFrameCount);
            if (pinFrameCount && available.Count < frameCount)
            {
                Logs.Warning($"[WanAnimate] The driving video is short: {available.Count} frame(s) left at offset {frameOffset} "
                    + $"for a {frameCount}-frame chunk — the tail of this chunk repeats the last driving frame.");
            }
            drivingFrames = FitFrames(available, frameCount);
            drivingFps = decoded.Fps > 0.5 ? (int)Math.Round(decoded.Fps) : null;
            Logs.Info($"[WanAnimate] driving video decoded: {decodedCount} frame(s) @ {decoded.Fps:0.##} fps → "
                + $"{frameCount} frame(s) at {width}x{height}{(frameOffset > 0 ? $" from offset {frameOffset}" : "")}.");
        }

        bool needsAuto = request.DrivingAutoPreprocess && drivingFrames is not null
            && (request.DrivingPoseVideo is null || request.DrivingFaceVideo is null);
        YoloPosePipeline? pose = needsAuto ? TryCreatePosePipeline(backend) : null;
        try
        {
            Tensor poseClip = BuildPoseClip(backend, request, pose, drivingFrames, width, height, frameCount, frameOffset, cancel);
            try
            {
                Tensor faceClip = BuildFaceClip(backend, request, pose, drivingFrames, width, height, motionSize, frameCount, frameOffset, cancel);
                // Replacement-mode extras: background clip in pose-clip form; the character mask decodes as RGB
                // (channel 0 is read pipeline-side) and a single-frame mask is legal (it repeats there).
                Tensor? backgroundClip = null;
                Tensor? maskClip = null;
                try
                {
                    if (request.DrivingBackgroundVideo is not null)
                    {
                        backgroundClip = DecodeBackgroundClip(request.DrivingBackgroundVideo, width, height, frameCount, frameOffset, cancel);
                        Logs.Info(backgroundClip is null
                            ? $"[WanAnimate] background video is exhausted at offset {frameOffset} — this chunk's conditioning stays mid-gray."
                            : $"[WanAnimate] background video decoded to {backgroundClip.Shape[2]} frame(s) at {width}x{height}.");
                    }
                    if (request.DrivingMaskVideo is not null)
                    {
                        FfmpegProcessDecoder.Result decodedMask = DecodeClip(request.DrivingMaskVideo, width, height, frameOffset + frameCount, cancel);
                        // A single-frame mask bypasses the offset entirely (it repeats over every chunk); a multi-frame
                        // one is seeked, and a seek past its end drops the mask for this chunk.
                        List<byte[]> maskFrames = decodedMask.Frames.Count == 1
                            ? decodedMask.Frames
                            : DropLeadingFrames(decodedMask.Frames, frameOffset);
                        if (maskFrames.Count == 0)
                        {
                            Logs.Info($"[WanAnimate] character mask is exhausted at offset {frameOffset} — this chunk generates unmasked.");
                        }
                        else
                        {
                            if (maskFrames.Count >= frameCount)
                            {
                                FitFrames(maskFrames, frameCount);   // short (e.g. single-frame) masks repeat pipeline-side
                            }
                            maskClip = VideoRecipeUtils.PackRgbFramesToClip(maskFrames, width, height);
                            Logs.Info($"[WanAnimate] character mask decoded: {maskFrames.Count} frame(s) at {width}x{height}.");
                        }
                    }
                }
                catch
                {
                    backgroundClip?.Dispose();
                    maskClip?.Dispose();
                    faceClip.Dispose();
                    throw;
                }
                return new ResolvedClips(poseClip, faceClip, frameCount, drivingFps, backgroundClip, maskClip);
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

    /// <summary>Per-chunk frame count. A continuation chunk PINS the request's count: the latent geometry, noise shape
    /// and motion-prefix arithmetic were all fixed against it, so shrinking a late chunk because the seeked driving
    /// video ran short would break the sequence — upstream never shrinks either, it repeat-pads the pose's last
    /// frame. Chunk 0 keeps the shrink rule so single-chunk generation is unchanged.</summary>
    internal static int ResolveChunkFrames(int requestedFrames, int availableFrames, int temporalStep, bool pinFrameCount) =>
        pinFrameCount ? requestedFrames : ResolveDrivingFrames(requestedFrames, availableFrames, temporalStep);

    /// <summary>Drops the leading <paramref name="frameOffset"/> frames in place (ffmpeg is decoded from frame 0 —
    /// <see cref="FfmpegProcessDecoder"/> has no seek), leaving an empty list when the clip is shorter.</summary>
    internal static List<byte[]> DropLeadingFrames(List<byte[]> frames, int frameOffset)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frameOffset <= 0)
        {
            return frames;
        }
        frames.RemoveRange(0, Math.Min(frameOffset, frames.Count));
        return frames;
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
        List<byte[]>? drivingFrames, int width, int height, int frameCount, int frameOffset, CancellationToken cancel)
    {
        if (request.DrivingPoseVideo is not null)
        {
            Logs.Info("[WanAnimate] pose branch: using the supplied pre-rendered pose video.");
            return DecodeToClip(request.DrivingPoseVideo, width, height, frameCount, frameOffset, "pose", cancel);
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
        List<byte[]>? drivingFrames, int width, int height, int motionSize, int frameCount, int frameOffset,
        CancellationToken cancel)
    {
        // One face crop per pose frame, as upstream (nodes_wan.py face_video[:length], and the reference
        // preprocessor emits a crop per driving frame). Taking frameCount-1 left the last latent frame's
        // motion vector zeroed; CausalConv1d replicate-pads on the left, so indices still line up and only
        // the tail was affected -- which is why it survived earlier validation.
        int faceFrames = frameCount;
        if (request.DrivingFaceVideo is not null)
        {
            Logs.Info("[WanAnimate] face branch: using the supplied pre-cropped face video.");
            return DecodeToClip(request.DrivingFaceVideo, motionSize, motionSize, faceFrames, frameOffset, "face", cancel);
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

    /// <summary>Decodes an override clip at the exact target geometry from <paramref name="frameOffset"/> onward and
    /// packs it, truncated/repeat-padded to <paramref name="numFrames"/> (the extension's <c>DecodeControlClip</c>
    /// semantics).</summary>
    private static Tensor DecodeToClip(VideoClip clip, int width, int height, int numFrames, int frameOffset,
        string branch, CancellationToken cancel)
    {
        FfmpegProcessDecoder.Result decoded = DecodeClip(clip, width, height, frameOffset + numFrames, cancel);
        int decodedCount = decoded.Frames.Count;
        List<byte[]> frames = DropLeadingFrames(decoded.Frames, frameOffset);
        if (frames.Count == 0)
        {
            throw new InvalidOperationException(
                $"The Wan-Animate {branch} clip ran out at frame {frameOffset} (only {decodedCount} frame(s) decoded) — "
                + "lower AnimateTotalFrames or supply a longer clip.");
        }
        return VideoRecipeUtils.PackRgbFramesToClip(FitFrames(frames, numFrames), width, height);
    }

    /// <summary>Background clip for one chunk, seeked and truncated but NEVER repeat-padded: upstream writes background
    /// only up to the decoded length and leaves the rest of the conditioning mid-gray, and a seek past its end drops
    /// the background for that chunk entirely (null).</summary>
    private static Tensor? DecodeBackgroundClip(VideoClip clip, int width, int height, int numFrames, int frameOffset,
        CancellationToken cancel)
    {
        FfmpegProcessDecoder.Result decoded = DecodeClip(clip, width, height, frameOffset + numFrames, cancel);
        List<byte[]> frames = DropLeadingFrames(decoded.Frames, frameOffset);
        if (frames.Count == 0)
        {
            return null;
        }
        if (frames.Count > numFrames)
        {
            frames.RemoveRange(numFrames, frames.Count - numFrames);
        }
        return VideoRecipeUtils.PackRgbFramesToClip(frames, width, height);
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
