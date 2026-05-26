using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Cpu;
using SharpInference.Tests.Common;
using SharpInference.Vision.Codec;
using SharpInference.Vision.Detection;
using Xunit;
using Xunit.Abstractions;

namespace SharpInference.Vision.Tests;

/// <summary>YOLO end-to-end test on a real photograph — Ultralytics' standard <c>bus.jpg</c>
/// (converted to PNG for our pure-C# decoder). The image contains a bus and several pedestrians,
/// so a working YOLOv8n COCO model should fire on at least one of those classes with confidence > 0.25.</summary>
public sealed class YoloRealImageTest
{
    private readonly ITestOutputHelper _output;

    public YoloRealImageTest(ITestOutputHelper output) => _output = output;

    private static string TestImagePath =>
        Path.Combine(AppContext.BaseDirectory, "TestData", "bus.png");

    [Fact]
    public void YoloV11n_DetectsPersonOrBus_OnUltralyticsBusImage()
    {
        string checkpoint = TestPaths.Yolo.V11nFolded;
        if (!File.Exists(checkpoint))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {checkpoint}");
            return;
        }
        if (!File.Exists(TestImagePath))
        {
            _output.WriteLine($"SKIPPED: Test image not found: {TestImagePath}");
            return;
        }

        Stopwatch sw = Stopwatch.StartNew();
        (byte[] rgb, int width, int height) = PngDecoder.DecodeFromFile(TestImagePath);
        _output.WriteLine($"[png decode] {sw.ElapsedMilliseconds} ms — {width}x{height}");

        using IBackend backend = new CpuBackend();
        using YoloPipeline pipeline = YoloPipeline.LoadV11(backend, YoloConfig.YoloV11n, checkpoint, inputSize: 640);

        sw.Restart();
        IReadOnlyList<YoloDetection> dets = pipeline.Detect(rgb, width, height,
            confidenceThreshold: 0.25f, iouThreshold: 0.45f);
        _output.WriteLine($"[YOLO11n detect] {sw.ElapsedMilliseconds} ms — {dets.Count} detections");

        for (int i = 0; i < dets.Count; i++)
        {
            YoloDetection d = dets[i];
            _output.WriteLine($"  [{i}] {pipeline.GetLabel(d.ClassId)} (class {d.ClassId}) conf={d.Confidence:F3} box=({d.X1:F0},{d.Y1:F0})→({d.X2:F0},{d.Y2:F0})");
        }

        Assert.True(dets.Count > 0, "YOLO11n produced no detections on bus.png — model or post-processor may be broken.");
        bool foundPersonOrBus = dets.Any(d => d.ClassId == 0 || d.ClassId == 5);
        Assert.True(foundPersonOrBus,
            $"Expected at least one person (class 0) or bus (class 5) detection. Got: {string.Join(", ", dets.Select(d => pipeline.GetLabel(d.ClassId)))}");

        foreach (YoloDetection d in dets)
        {
            Assert.InRange(d.X1, 0f, width);
            Assert.InRange(d.Y1, 0f, height);
            Assert.InRange(d.X2, 0f, width);
            Assert.InRange(d.Y2, 0f, height);
            Assert.True(d.X2 > d.X1 && d.Y2 > d.Y1, $"Degenerate box: {d}");
        }
    }

    [Fact]
    public void YoloV8n_DetectsPersonOrBus_OnUltralyticsBusImage()
    {
        string checkpoint = TestPaths.Yolo.V8nFolded;
        if (!File.Exists(checkpoint))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {checkpoint}");
            return;
        }
        if (!File.Exists(TestImagePath))
        {
            _output.WriteLine($"SKIPPED: Test image not found: {TestImagePath}");
            return;
        }

        Stopwatch sw = Stopwatch.StartNew();
        (byte[] rgb, int width, int height) = PngDecoder.DecodeFromFile(TestImagePath);
        _output.WriteLine($"[png decode] {sw.ElapsedMilliseconds} ms — {width}x{height}");

        using IBackend backend = new CpuBackend();
        using YoloPipeline pipeline = new(backend, YoloConfig.YoloV8n, checkpoint, inputSize: 640);

        sw.Restart();
        IReadOnlyList<YoloDetection> dets = pipeline.Detect(rgb, width, height,
            confidenceThreshold: 0.25f, iouThreshold: 0.45f);
        _output.WriteLine($"[detect] {sw.ElapsedMilliseconds} ms — {dets.Count} detections");

        // Log every detection so the test output makes failure modes obvious.
        for (int i = 0; i < dets.Count; i++)
        {
            YoloDetection d = dets[i];
            _output.WriteLine($"  [{i}] {pipeline.GetLabel(d.ClassId)} (class {d.ClassId}) conf={d.Confidence:F3} box=({d.X1:F0},{d.Y1:F0})→({d.X2:F0},{d.Y2:F0})");
        }

        // bus.jpg is THE Ultralytics smoke-test image — a bus with four people in front. A working
        // YOLOv8n produces 1 bus + 4 persons with high confidence. We assert the bare minimum:
        // at least one detection, with at least one being either "person" (class 0) or "bus" (class 5).
        Assert.True(dets.Count > 0, "Expected at least one detection on bus.jpg — model or post-processor may be broken.");
        bool foundPersonOrBus = dets.Any(d => d.ClassId == 0 || d.ClassId == 5);
        Assert.True(foundPersonOrBus,
            $"Expected at least one person (class 0) or bus (class 5) detection on bus.jpg. Got: {string.Join(", ", dets.Select(d => pipeline.GetLabel(d.ClassId)))}");

        // Sanity-check coordinates are in image-space.
        foreach (YoloDetection d in dets)
        {
            Assert.InRange(d.X1, 0f, width);
            Assert.InRange(d.Y1, 0f, height);
            Assert.InRange(d.X2, 0f, width);
            Assert.InRange(d.Y2, 0f, height);
            Assert.True(d.X2 > d.X1 && d.Y2 > d.Y1, $"Degenerate box: {d}");
        }
    }
}
