using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>Builds Wan-Animate's pose driving clip by running YOLO11-pose per driving frame and rendering an
/// OpenPose-18 skeleton (<see cref="OpenPoseRenderer"/>) — the DWPose/ControlNet convention the checkpoint's
/// <c>pose_patch_embedding</c> was trained on. Feeding the raw driving RGB instead is out-of-distribution and
/// weakens motion following. Ported from the SwarmUI extension's <c>WanAnimatePosePreprocessor</c>.</summary>
internal static class WanAnimatePoseClipBuilder
{
    /// <summary>Returns the <c>[1, 3, T, H, W]</c> pose-skeleton clip in [-1, 1] (black background, colored OpenPose
    /// limbs/joints) for <c>T = frames.Count</c>; frames are interleaved HWC RGB24 at <paramref name="width"/>×<paramref name="height"/>.</summary>
    internal static Tensor Build(IBackend backend, YoloPosePipeline pose, IReadOnlyList<byte[]> frames,
        int width, int height, CancellationToken cancel)
    {
        int numFrames = frames.Count;
        byte[][] skeletons = new byte[numFrames][];
        backend.PreloadWeights(pose.EnumerateWeights());
        int rendered = 0;
        try
        {
            for (int f = 0; f < numFrames; f++)
            {
                cancel.ThrowIfCancellationRequested();
                IReadOnlyList<PoseDetection> people = pose.Detect(frames[f], width, height, confidenceThreshold: 0.25f, iouThreshold: 0.45f);
                backend.FreeActivations();
                skeletons[f] = OpenPoseRenderer.RenderBodyPose(people, width, height);
                if (people.Count > 0)
                {
                    rendered++;
                }
            }
        }
        finally
        {
            backend.Sync();
            backend.FreeWeights(pose.EnumerateWeights());
        }
        Logs.Info($"[WanAnimate] pose preprocess: {rendered}/{numFrames} frames skeleton-rendered → {width}x{height} pose clip.");
        return VideoRecipeUtils.PackRgbFramesToClip(skeletons, width, height);
    }
}
