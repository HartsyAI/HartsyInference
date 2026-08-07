using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection;
using HartsyInference.Vision.FaceDetection;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Builds Wan-Animate's face-motion driving clip by localizing the face per driving frame (YOLO11-pose
/// keypoints → <see cref="PoseFaceCrop"/>) and bilinearly resampling a face-centered square to the motion-encoder
/// resolution; a null pose pipeline (or an undetected frame) center-crop-squares instead. Ported from the SwarmUI
/// extension's <c>WanAnimateFacePreprocessor</c>.</summary>
internal static class WanAnimateFaceClipBuilder
{
    /// <summary>Mid-gray sample for crop pixels outside the frame (matches the YOLO letterbox fill).</summary>
    internal const float OutOfFramePad = 114f;

    /// <summary>Returns the <c>[1, 3, T, S, S]</c> face clip in [-1, 1] for <c>T = frames.Count</c>, <c>S = motionSize</c>;
    /// frames are interleaved HWC RGB24 at <paramref name="width"/>×<paramref name="height"/> (where pose runs).</summary>
    internal static unsafe Tensor Build(IBackend backend, YoloPosePipeline? pose, IReadOnlyList<byte[]> frames,
        int width, int height, int motionSize, CancellationToken cancel)
    {
        int faceFrames = frames.Count;
        Tensor clip = new Tensor(new TensorShape([1L, 3, faceFrames, motionSize, motionSize]), DType.F32);
        float* cp = (float*)clip.DataPointer;
        long framePix = (long)motionSize * motionSize;
        long perChannel = faceFrames * framePix;

        if (pose is not null)
        {
            backend.PreloadWeights(pose.EnumerateWeights());
        }
        int detected = 0, fallbacks = 0;
        try
        {
            for (int f = 0; f < faceFrames; f++)
            {
                cancel.ThrowIfCancellationRequested();
                byte[] rgb = frames[f];
                IReadOnlyList<PoseDetection>? people = null;
                if (pose is not null)
                {
                    people = pose.Detect(rgb, width, height, confidenceThreshold: 0.25f, iouThreshold: 0.45f);
                    backend.FreeActivations();
                }
                PoseFaceCrop.Rect crop;
                if (people is { Count: > 0 })
                {
                    PoseDetection best = people[0];
                    for (int i = 1; i < people.Count; i++)
                    {
                        if (people[i].Confidence > best.Confidence)
                        {
                            best = people[i];
                        }
                    }
                    crop = PoseFaceCrop.ComputeSquareCrop(best, width, height);
                    detected++;
                }
                else
                {
                    crop = CenterSquareCrop(width, height);
                    fallbacks++;
                }
                float[] chw = SampleSquareChw(rgb, width, height, crop, motionSize);
                for (int c = 0; c < 3; c++)
                {
                    chw.AsSpan(c * (int)framePix, (int)framePix)
                        .CopyTo(new Span<float>(cp + c * perChannel + f * framePix, (int)framePix));
                }
            }
        }
        catch
        {
            clip.Dispose();
            throw;
        }
        finally
        {
            if (pose is not null)
            {
                backend.Sync();
                backend.FreeWeights(pose.EnumerateWeights());
            }
        }
        if (pose is not null)
        {
            Logs.Info($"[WanAnimate] face preprocess: {detected}/{faceFrames} frames face-detected"
                + $"{(fallbacks > 0 ? $", {fallbacks} center-crop fallback" : "")} → {motionSize}² face clip.");
        }
        return clip;
    }

    /// <summary>Centered square of the frame's shorter side — the no-detection / no-detector crop.</summary>
    internal static PoseFaceCrop.Rect CenterSquareCrop(int width, int height)
    {
        float side = Math.Min(width, height);
        return new PoseFaceCrop.Rect((width - side) * 0.5f, (height - side) * 0.5f, side);
    }

    /// <summary>Bilinearly samples the source frame's square crop into a <c>[3, outSize, outSize]</c> CHW buffer in
    /// [-1, 1]; samples outside the frame read mid-gray (<see cref="OutOfFramePad"/>) so an off-image crop pads cleanly.</summary>
    internal static float[] SampleSquareChw(byte[] rgb, int width, int height, PoseFaceCrop.Rect crop, int outSize)
    {
        float[] chw = new float[3 * outSize * outSize];
        int framePix = outSize * outSize;
        float step = crop.Size / outSize;
        for (int oy = 0; oy < outSize; oy++)
        {
            float sy = crop.Y + (oy + 0.5f) * step - 0.5f;
            int y0 = (int)MathF.Floor(sy);
            float fy = sy - y0;
            for (int ox = 0; ox < outSize; ox++)
            {
                float sx = crop.X + (ox + 0.5f) * step - 0.5f;
                int x0 = (int)MathF.Floor(sx);
                float fx = sx - x0;
                int outPix = oy * outSize + ox;
                for (int c = 0; c < 3; c++)
                {
                    float v00 = Sample(rgb, width, height, x0, y0, c);
                    float v10 = Sample(rgb, width, height, x0 + 1, y0, c);
                    float v01 = Sample(rgb, width, height, x0, y0 + 1, c);
                    float v11 = Sample(rgb, width, height, x0 + 1, y0 + 1, c);
                    float top = v00 + (v10 - v00) * fx;
                    float bot = v01 + (v11 - v01) * fx;
                    float val = top + (bot - top) * fy;   // [0, 255]
                    chw[c * framePix + outPix] = val / 127.5f - 1f;
                }
            }
        }
        return chw;
    }

    private static float Sample(byte[] src, int w, int h, int x, int y, int c) =>
        x < 0 || y < 0 || x >= w || y >= h ? OutOfFramePad : src[((long)y * w + x) * 3 + c];
}
