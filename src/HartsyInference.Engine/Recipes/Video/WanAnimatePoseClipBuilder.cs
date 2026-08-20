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
        // Wan2.2's renderer, not controlnet_aux's: line width is canvas-relative, limbs overwrite, one subject,
        // and joints below 0.5 are dropped. The detector stays at 0.25 — upstream's person detector is lower still.
        OpenPoseRenderer.Profile profile = OpenPoseRenderer.Profile.Wan22Animate with
        {
            LineWidth = OpenPoseRenderer.Wan22LineWidth(width, height),
        };
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
                skeletons[f] = OpenPoseRenderer.RenderBodyPose(people, width, height, 1f, 1f, profile);
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
        DumpSkeletons(skeletons, width, height);
        return VideoRecipeUtils.PackRgbFramesToClip(skeletons, width, height);
    }

    /// <summary>Writes the rendered skeletons as PPMs under <c>HARTSY_ANIMATE_POSE_DUMP</c>. The pose clip is the one
    /// driving input nothing else can show you — it is VAE-encoded straight into the latent, so a bad render reads as
    /// a model failure.</summary>
    private static void DumpSkeletons(byte[][] skeletons, int width, int height)
    {
        string? dir = Environment.GetEnvironmentVariable("HARTSY_ANIMATE_POSE_DUMP");
        if (string.IsNullOrEmpty(dir)) return;
        try
        {
            Directory.CreateDirectory(dir);
            for (int f = 0; f < skeletons.Length; f++)
            {
                using FileStream fs = File.Create(Path.Combine(dir, $"pose_{f:D4}.ppm"));
                byte[] header = System.Text.Encoding.ASCII.GetBytes($"P6\n{width} {height}\n255\n");
                fs.Write(header, 0, header.Length);
                fs.Write(skeletons[f], 0, skeletons[f].Length);
            }
            Logs.Info($"[WanAnimate] pose dump: {skeletons.Length} skeleton frame(s) → '{dir}'.");
        }
        catch (Exception error)
        {
            Logs.Warning($"[WanAnimate] pose dump failed: {error.Message}");
        }
    }
}
