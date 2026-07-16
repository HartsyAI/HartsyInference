using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Detection;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Unit tests for the OpenPose ControlNet preprocessor's render path — synthetic keypoints through
/// <see cref="OpenPosePreprocessor.RenderToTensor"/>, no weights or GPU required.</summary>
public sealed class OpenPosePreprocessorTests
{
    /// <summary>A standing figure in a 640×480 source image, all 17 COCO keypoints confident.</summary>
    internal static PoseDetection MakeStandingFigure(float conf = 0.9f)
    {
        Keypoint[] kpts =
        [
            new(320, 80, conf),    // nose
            new(310, 70, conf),    // left eye
            new(330, 70, conf),    // right eye
            new(300, 75, conf),    // left ear
            new(340, 75, conf),    // right ear
            new(280, 140, conf),   // left shoulder
            new(360, 140, conf),   // right shoulder
            new(260, 220, conf),   // left elbow
            new(380, 220, conf),   // right elbow
            new(250, 300, conf),   // left wrist
            new(390, 300, conf),   // right wrist
            new(295, 300, conf),   // left hip
            new(345, 300, conf),   // right hip
            new(290, 380, conf),   // left knee
            new(350, 380, conf),   // right knee
            new(285, 460, conf),   // left ankle
            new(355, 460, conf),   // right ankle
        ];
        return new PoseDetection(new YoloDetection(250, 60, 390, 470, conf, 0), kpts);
    }

    [Fact]
    public unsafe void RenderToTensor_SyntheticPose_ProducesSkeletonInRange()
    {
        PoseDetection person = MakeStandingFigure();
        using Tensor cond = OpenPosePreprocessor.RenderToTensor([person], 640, 480, 640, 480);

        Assert.Equal(new TensorShape(1, 3, 480, 640), cond.Shape);
        Assert.Equal(DType.F32, cond.DType);

        float* p = (float*)cond.DataPointer;
        long n = cond.Shape.ElementCount;
        double sum = 0;
        for (long i = 0; i < n; i++)
        {
            Assert.InRange(p[i], 0f, 1f);
            sum += p[i];
        }
        Assert.True(sum > 0, "skeleton render is empty");

        // Corners are far from every limb — background must stay black.
        int plane = 640 * 480;
        for (int c = 0; c < 3; c++)
        {
            Assert.Equal(0f, p[c * plane]);
            Assert.Equal(0f, p[c * plane + 639]);
            Assert.Equal(0f, p[(c + 1) * plane - 1]);
        }
    }

    [Fact]
    public unsafe void RenderToTensor_NoPeople_AllBlack()
    {
        using Tensor cond = OpenPosePreprocessor.RenderToTensor([], 640, 480, 512, 512);
        Assert.Equal(new TensorShape(1, 3, 512, 512), cond.Shape);
        float* p = (float*)cond.DataPointer;
        for (long i = 0; i < cond.Shape.ElementCount; i++)
            Assert.Equal(0f, p[i]);
    }

    [Fact]
    public unsafe void RenderToTensor_ScalesKeypointsToOutputResolution()
    {
        PoseDetection person = MakeStandingFigure();
        using Tensor cond = OpenPosePreprocessor.RenderToTensor([person], 640, 480, 320, 240);

        Assert.Equal(new TensorShape(1, 3, 240, 320), cond.Shape);
        float* p = (float*)cond.DataPointer;

        // Neck (shoulder midpoint) is at (320, 140) in source pixels → (160, 70) at half resolution.
        // A joint dot (radius 4) must land there; probe the max over a small window around it.
        float peak = 0f;
        for (int y = 66; y <= 74; y++)
            for (int x = 156; x <= 164; x++)
                for (int c = 0; c < 3; c++)
                    peak = MathF.Max(peak, p[c * 320 * 240 + y * 320 + x]);
        Assert.True(peak > 0.2f, $"no skeleton content near scaled neck position (peak={peak})");

        // Nothing should be drawn at unscaled source coordinates that fall outside the figure's scaled
        // footprint — e.g. (320, 140) in OUTPUT pixels maps back to (640, 280) source, outside the body.
        float offBody = 0f;
        for (int c = 0; c < 3; c++)
            offBody = MathF.Max(offBody, p[c * 320 * 240 + 140 * 320 + 316]);
        Assert.Equal(0f, offBody);
    }

    [Fact]
    public unsafe void RenderToTensor_LowConfidenceKeypoints_Skipped()
    {
        PoseDetection person = MakeStandingFigure(conf: 0.05f);
        using Tensor cond = OpenPosePreprocessor.RenderToTensor([person], 640, 480, 640, 480, visThreshold: 0.3f);
        float* p = (float*)cond.DataPointer;
        for (long i = 0; i < cond.Shape.ElementCount; i++)
            Assert.Equal(0f, p[i]);
    }
}
