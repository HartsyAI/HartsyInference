using Xunit;
using HartsyInference.Vision.Detection;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pins the Wan2.2 render conventions against controlnet_aux's. A skeleton drawn for the wrong one is
/// out of distribution for whatever consumes it, and the two profiles are the only thing keeping them apart.</summary>
public class WanAnimatePoseProfileTests
{
    [Theory]
    [InlineData(512, 512, 1)]     // int(512/200) - 1 = 1
    [InlineData(720, 1280, 2)]    // int(720/200) - 1 = 2
    [InlineData(1080, 1920, 4)]   // int(1080/200) - 1 = 4
    [InlineData(128, 128, 1)]     // floors at 1, never 0
    public void Wan22LineWidthIsCanvasRelative(int width, int height, int expected) =>
        Assert.Equal(expected, OpenPoseRenderer.Wan22LineWidth(width, height));

    [Fact]
    public void ProfilesDifferOnEveryConventionThatMatters()
    {
        OpenPoseRenderer.Profile aux = OpenPoseRenderer.Profile.ControlNetAux;
        OpenPoseRenderer.Profile wan = OpenPoseRenderer.Profile.Wan22Animate;
        Assert.Equal(0.3f, aux.VisThreshold);
        Assert.Equal(0.5f, wan.VisThreshold);
        Assert.False(aux.LargestPersonOnly);
        Assert.True(wan.LargestPersonOnly);
        Assert.True(wan.OverwriteLimbs);
        Assert.True(wan.AverageNeckConfidence);
        Assert.False(aux.AverageNeckConfidence);
    }

    /// <summary>A weak shoulder must not delete the neck under Wan's rule — it anchors 5 of the limbs, so dropping
    /// it guts the skeleton. 0.8 and 0.3 average to 0.55, clearing 0.5, while the strict rule rejects on the 0.3.</summary>
    [Fact]
    public void AveragedNeckSurvivesOneWeakShoulder()
    {
        byte[] wan = Render(OpenPoseRenderer.Profile.Wan22Animate with { LineWidth = 4 }, 0.8f, 0.3f);
        byte[] strict = Render(OpenPoseRenderer.Profile.Wan22Animate with { LineWidth = 4, AverageNeckConfidence = false },
            0.8f, 0.3f);
        Assert.True(Ink(wan) > Ink(strict), "averaging the shoulder confidences must keep the neck-anchored limbs");
    }

    /// <summary>The width fix is the whole point: a fixed 4 at 512 inks far more of the canvas than upstream's 1.</summary>
    [Fact]
    public void CanvasRelativeWidthInksFarLessThanAFixedFour()
    {
        byte[] thin = Render(OpenPoseRenderer.Profile.Wan22Animate with { LineWidth = 1 }, 0.9f, 0.9f);
        byte[] thick = Render(OpenPoseRenderer.Profile.Wan22Animate with { LineWidth = 4 }, 0.9f, 0.9f);
        Assert.True(Ink(thick) > Ink(thin) * 2, $"expected the fixed-4 render to ink far more; {Ink(thick)} vs {Ink(thin)}");
    }

    private static byte[] Render(OpenPoseRenderer.Profile profile, float leftShoulder, float rightShoulder)
    {
        Keypoint[] kpts = new Keypoint[17];
        for (int i = 0; i < kpts.Length; i++) kpts[i] = new Keypoint(0f, 0f, 0f);
        kpts[5] = new Keypoint(200f, 200f, leftShoulder);    // COCO L-shoulder
        kpts[6] = new Keypoint(300f, 200f, rightShoulder);   // COCO R-shoulder
        kpts[11] = new Keypoint(210f, 350f, 0.9f);           // L-hip, so a neck-anchored limb exists
        kpts[12] = new Keypoint(290f, 350f, 0.9f);           // R-hip
        PoseDetection person = new(new YoloDetection(150f, 150f, 350f, 400f, 0.9f, 0), kpts);
        return OpenPoseRenderer.RenderBodyPose([person], 512, 512, 1f, 1f, profile);
    }

    private static int Ink(byte[] canvas)
    {
        int n = 0;
        for (int i = 0; i < canvas.Length; i += 3)
            if (canvas[i] != 0 || canvas[i + 1] != 0 || canvas[i + 2] != 0) n++;
        return n;
    }
}
