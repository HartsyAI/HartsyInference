using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Vision.Codec;
using HartsyInference.Vision.Detection;
using HartsyInference.Vision.FaceDetection;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests;

/// <summary>YOLO11-pose end-to-end on a real photo — env-gated so it runs only when the folded checkpoint +
/// image are supplied (<c>HARTSY_POSE_CKPT</c>, <c>HARTSY_POSE_IMG</c>). Prints the person box + 17 keypoints so
/// the caller can diff against an Ultralytics reference (<c>tools</c> convert + <c>model.predict</c>).</summary>
[Trait("Category", "Integration")]
public sealed class YoloPoseRealImageTest
{
    private readonly ITestOutputHelper _output;

    public YoloPoseRealImageTest(ITestOutputHelper output) => _output = output;

    private static readonly string[] KptNames =
    [
        "nose", "L-eye", "R-eye", "L-ear", "R-ear", "L-shoulder", "R-shoulder", "L-elbow", "R-elbow",
        "L-wrist", "R-wrist", "L-hip", "R-hip", "L-knee", "R-knee", "L-ankle", "R-ankle",
    ];

    [Fact]
    public void YoloV11nPose_DetectsPersonKeypoints()
    {
        string? ckpt = Environment.GetEnvironmentVariable("HARTSY_POSE_CKPT");
        string? img = Environment.GetEnvironmentVariable("HARTSY_POSE_IMG");
        if (string.IsNullOrEmpty(ckpt) || !File.Exists(ckpt)) { _output.WriteLine($"SKIPPED: HARTSY_POSE_CKPT missing ({ckpt})"); return; }
        if (string.IsNullOrEmpty(img) || !File.Exists(img)) { _output.WriteLine($"SKIPPED: HARTSY_POSE_IMG missing ({img})"); return; }

        (byte[] rgb, int width, int height) = PngDecoder.DecodeFromFile(img);
        _output.WriteLine($"image {width}x{height}");

        using IBackend backend = new CpuBackend();
        using YoloPosePipeline pipeline = new(backend, YoloConfig.YoloV11nPose, ckpt, inputSize: 640);
        IReadOnlyList<PoseDetection> people = pipeline.Detect(rgb, width, height, confidenceThreshold: 0.25f, iouThreshold: 0.45f);

        _output.WriteLine($"persons: {people.Count}");
        Assert.NotEmpty(people);
        for (int p = 0; p < people.Count; p++)
        {
            PoseDetection person = people[p];
            YoloDetection b = person.Box;
            _output.WriteLine($"[{p}] conf={person.Confidence:F3} box=({b.X1:F1},{b.Y1:F1})→({b.X2:F1},{b.Y2:F1})");
            Assert.Equal(17, person.Keypoints.Count);
            for (int k = 0; k < person.Keypoints.Count; k++)
            {
                Keypoint kp = person.Keypoints[k];
                _output.WriteLine($"    {KptNames[k],-11} ({kp.X:F1}, {kp.Y:F1})  v={kp.Confidence:F2}");
            }
            PoseFaceCrop.Rect crop = PoseFaceCrop.ComputeSquareCrop(person, width, height);
            _output.WriteLine($"    FACECROP x={crop.X:F1} y={crop.Y:F1} size={crop.Size:F1}");
        }

        // Render the OpenPose skeleton and dump raw rgb24 (a sibling script converts + overlays for a visual check).
        string? skelOut = Environment.GetEnvironmentVariable("HARTSY_POSE_SKELETON_OUT");
        if (!string.IsNullOrEmpty(skelOut))
        {
            byte[] skel = OpenPoseRenderer.RenderBodyPose(people, width, height);
            File.WriteAllBytes(skelOut, skel);
            _output.WriteLine($"    SKELETON {width}x{height} rgb24 -> {skelOut}");
        }
    }
}
