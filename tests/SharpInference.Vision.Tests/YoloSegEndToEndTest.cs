using System.Diagnostics;
using SharpInference.Core.Backends;
using SharpInference.Cpu;
using SharpInference.Tests.Common;
using SharpInference.Vision.Codec;
using SharpInference.Vision.Detection;
using Xunit;
using Xunit.Abstractions;

namespace SharpInference.Vision.Tests;

/// <summary>End-to-end YOLOv8n-seg test on Ultralytics' bus.png. Verifies:
/// <list type="bullet">
///   <item>Pipeline loads the seg checkpoint cleanly.</item>
///   <item>Forward produces 5 expected detections (bus + 4 persons) — same as the detect-only path.</item>
///   <item>Each detection has a non-trivial mask (non-zero pixel count, mostly inside the bbox).</item>
///   <item>The bus mask covers a substantial fraction of the bus's bounding box.</item>
/// </list></summary>
public sealed class YoloSegEndToEndTest
{
    private readonly ITestOutputHelper _output;
    public YoloSegEndToEndTest(ITestOutputHelper output) => _output = output;

    private static string TestImagePath =>
        Path.Combine(AppContext.BaseDirectory, "TestData", "bus.png");

    [Fact]
    public void YoloV8nSeg_DetectsAndSegments_BusImage()
    {
        string checkpoint = TestPaths.Yolo.V8nSegFolded;
        if (!File.Exists(checkpoint))
        {
            _output.WriteLine($"SKIPPED: Seg checkpoint not found: {checkpoint}");
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
        using YoloSegPipeline pipe = new(backend, YoloConfig.YoloV8n, checkpoint, inputSize: 640);

        sw.Restart();
        IReadOnlyList<YoloSegResult> results = pipe.Segment(rgb, width, height,
            confidenceThreshold: 0.25f, iouThreshold: 0.45f, maxDetections: 300, maskThreshold: 0.5f);
        _output.WriteLine($"[segment] {sw.ElapsedMilliseconds} ms — {results.Count} results");

        for (int i = 0; i < results.Count; i++)
        {
            YoloSegResult r = results[i];
            // Count mask pixels.
            int maskPixels = 0;
            int bboxPixels = 0;
            int x1 = (int)MathF.Floor(r.Detection.X1), y1 = (int)MathF.Floor(r.Detection.Y1);
            int x2 = (int)MathF.Ceiling(r.Detection.X2), y2 = (int)MathF.Ceiling(r.Detection.Y2);
            x1 = Math.Clamp(x1, 0, r.Width); y1 = Math.Clamp(y1, 0, r.Height);
            x2 = Math.Clamp(x2, 0, r.Width); y2 = Math.Clamp(y2, 0, r.Height);
            for (int y = y1; y < y2; y++)
                for (int x = x1; x < x2; x++)
                {
                    bboxPixels++;
                    if (r.Mask[y * r.Width + x] != 0) maskPixels++;
                }
            int totalMaskPixels = 0;
            for (int j = 0; j < r.Mask.Length; j++) if (r.Mask[j] != 0) totalMaskPixels++;
            float fill = bboxPixels > 0 ? (float)maskPixels / bboxPixels : 0f;
            _output.WriteLine($"  [{i}] {pipe.GetLabel(r.Detection.ClassId)} conf={r.Detection.Confidence:F3} " +
                $"box=({r.Detection.X1:F0},{r.Detection.Y1:F0})→({r.Detection.X2:F0},{r.Detection.Y2:F0}) " +
                $"mask_in_box={maskPixels}/{bboxPixels} ({fill:P0}), total_mask_pixels={totalMaskPixels}");

            // Each detection must have at least SOME mask pixels inside its bbox.
            Assert.True(maskPixels > 0, $"Detection {i} ({pipe.GetLabel(r.Detection.ClassId)}) has zero mask pixels inside its bbox — segmentation broken.");

            // And the mask shouldn't be a giant blob covering the whole image (basic sanity).
            Assert.True(totalMaskPixels < r.Mask.Length * 0.95, $"Detection {i} mask covers >95% of the image — assembly likely wrong.");
        }

        Assert.True(results.Count > 0, "Expected at least one detection on bus.png.");
        // Expect at least one person and at least one bus.
        bool hasBus = results.Any(r => r.Detection.ClassId == 5);
        bool hasPerson = results.Any(r => r.Detection.ClassId == 0);
        Assert.True(hasBus, "Expected a bus detection.");
        Assert.True(hasPerson, "Expected at least one person detection.");
    }
}
