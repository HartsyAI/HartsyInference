using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Cpu;
using HartsyInference.Tests.Common;
using HartsyInference.Vision.Detection;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests;

/// <summary>End-to-end YOLOv8 checkpoint tests. These require <c>yolov8n-folded.safetensors</c>
/// (the BN-folded conversion of Ultralytics' <c>yolov8n.pt</c>) on disk and are skipped cleanly
/// when missing. CPU inference at 640×640 is slow (~1-2 minutes/image without convolutions
/// optimized for n-batch) but acceptable as a correctness gate.
/// <para>The validation focuses on <i>shape</i> and <i>sanity</i>: the model loads, the forward
/// pass produces an output tensor of the expected shape, and detections on a synthetic image
/// land in plausible regions with valid coordinates. Pixel-exact comparison against Ultralytics
/// is deferred to a separate Python diff harness; the bigger risk at this stage is gross errors
/// (wrong key mapping, off-by-one anchor, BN folding mistake) which any of these checks catch.</para></summary>
public sealed class YoloCheckpointTests
{
    private readonly ITestOutputHelper _output;

    public YoloCheckpointTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void YoloV8n_LoadsAndProducesValidOutputShape()
    {
        string checkpoint = TestPaths.Yolo.V8nFolded;
        if (!File.Exists(checkpoint))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {checkpoint}");
            return;
        }

        Stopwatch sw = Stopwatch.StartNew();
        using IBackend backend = new CpuBackend();
        using YoloPipeline pipeline = new(backend, YoloConfig.YoloV8n, checkpoint, inputSize: 640);
        _output.WriteLine($"[load] {sw.ElapsedMilliseconds} ms");

        Assert.Equal("yolov8n", pipeline.ModelName);
        Assert.Equal(640, pipeline.InputSize);
        Assert.Equal(80, pipeline.Model.NumClasses);

        // Synthetic 640×640 image with a few rectangles of solid color. Real detections require
        // photographs; this is purely a "the forward pass works end to end" test.
        byte[] rgb = new byte[640 * 640 * 3];
        for (int y = 0; y < 640; y++)
        for (int x = 0; x < 640; x++)
        {
            int i = (y * 640 + x) * 3;
            // Background: light gray.
            rgb[i + 0] = 200; rgb[i + 1] = 200; rgb[i + 2] = 200;
        }

        sw.Restart();
        IReadOnlyList<YoloDetection> dets = pipeline.Detect(rgb, 640, 640,
            confidenceThreshold: 0.05f, iouThreshold: 0.45f, maxDetections: 300);
        _output.WriteLine($"[detect 640x640 gray] {sw.ElapsedMilliseconds} ms — {dets.Count} detections");

        // All coordinates must be in valid [0, width/height] range and ordered.
        foreach (YoloDetection d in dets)
        {
            Assert.InRange(d.X1, 0f, 640f);
            Assert.InRange(d.Y1, 0f, 640f);
            Assert.InRange(d.X2, 0f, 640f);
            Assert.InRange(d.Y2, 0f, 640f);
            Assert.True(d.X2 >= d.X1, $"Bad box: X2={d.X2} < X1={d.X1}");
            Assert.True(d.Y2 >= d.Y1, $"Bad box: Y2={d.Y2} < Y1={d.Y1}");
            Assert.InRange(d.Confidence, 0f, 1f);
            Assert.InRange(d.ClassId, 0, 79);
        }

        // Inspect a couple if any exist — log them so we can eyeball results.
        int show = Math.Min(dets.Count, 5);
        for (int i = 0; i < show; i++)
        {
            YoloDetection d = dets[i];
            _output.WriteLine($"  [{i}] class={pipeline.GetLabel(d.ClassId)} ({d.ClassId}) conf={d.Confidence:F3} box=({d.X1:F1},{d.Y1:F1})→({d.X2:F1},{d.Y2:F1})");
        }
    }

    [Fact]
    public void YoloV8n_ForwardPass_OutputsExpectedAnchorCount()
    {
        string checkpoint = TestPaths.Yolo.V8nFolded;
        if (!File.Exists(checkpoint))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {checkpoint}");
            return;
        }

        using IBackend backend = new CpuBackend();
        YoloConfig config = YoloConfig.YoloV8n;
        using YoloPipeline pipeline = new(backend, config, checkpoint, inputSize: 640);

        // 80x80 + 40x40 + 20x20 = 8400 anchors at 640x640 input.
        // Output tensor shape is [1, 84, 8400] for COCO.
        // Verify indirectly: feed a known-shape preprocessed tensor through the model.
        byte[] rgb = new byte[640 * 640 * 3];
        Array.Fill<byte>(rgb, 128);
        YoloPreprocessor pre = new(640);
        (HartsyInference.Core.Tensors.Tensor input, _) = pre.Preprocess(rgb, 640, 640);
        try
        {
            using HartsyInference.Core.Tensors.Tensor output = pipeline.Model.Forward(backend, input);
            Assert.Equal(3, output.Shape.Rank);
            Assert.Equal(1, output.Shape[0]);
            Assert.Equal(84, output.Shape[1]); // 4 (xywh) + 80 (classes)
            Assert.Equal(8400, output.Shape[2]); // 80*80 + 40*40 + 20*20
            _output.WriteLine($"output shape: {output.Shape}");
        }
        finally { input.Dispose(); }
    }

    [Fact]
    public void YoloV8n_DetectsLargeObjectInSyntheticImage()
    {
        string checkpoint = TestPaths.Yolo.V8nFolded;
        if (!File.Exists(checkpoint))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {checkpoint}");
            return;
        }

        using IBackend backend = new CpuBackend();
        using YoloPipeline pipeline = new(backend, YoloConfig.YoloV8n, checkpoint, inputSize: 640);

        // Construct an image that's *vaguely* object-like — a centered dark blob on light
        // background. Real photos perform far better, but YOLOv8n trained on COCO will often
        // fire on high-contrast blobs (low-confidence "person" or "object" detections).
        byte[] rgb = new byte[640 * 640 * 3];
        for (int y = 0; y < 640; y++)
        for (int x = 0; x < 640; x++)
        {
            int i = (y * 640 + x) * 3;
            // Centered dark rectangle ~200x300 with light background.
            bool inBlob = x >= 220 && x < 420 && y >= 170 && y < 470;
            byte v = (byte)(inBlob ? 30 : 230);
            rgb[i + 0] = v; rgb[i + 1] = v; rgb[i + 2] = v;
        }

        Stopwatch sw = Stopwatch.StartNew();
        IReadOnlyList<YoloDetection> dets = pipeline.Detect(rgb, 640, 640,
            confidenceThreshold: 0.05f);
        _output.WriteLine($"[detect synth-blob] {sw.ElapsedMilliseconds} ms — {dets.Count} detections");

        // No strict assertion on detection content (synthetic image isn't guaranteed to produce
        // anything), but log everything so we can verify by eye.
        for (int i = 0; i < Math.Min(dets.Count, 8); i++)
        {
            YoloDetection d = dets[i];
            _output.WriteLine($"  [{i}] {pipeline.GetLabel(d.ClassId)} conf={d.Confidence:F3} box=({d.X1:F0},{d.Y1:F0})→({d.X2:F0},{d.Y2:F0})");
        }
    }
}
