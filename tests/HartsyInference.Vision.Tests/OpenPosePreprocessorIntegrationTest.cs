using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Vision.Detection;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vision.Tests;

/// <summary>End-to-end OpenPose preprocessor over the real yolo11n-pose checkpoint (<c>HARTSY_POSE_CKPT</c>) on a
/// generated humanoid test image — skips cleanly when the checkpoint is absent.</summary>
[Trait("Category", "Integration")]
public sealed class OpenPosePreprocessorIntegrationTest
{
    private readonly ITestOutputHelper _output;

    public OpenPosePreprocessorIntegrationTest(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void Process_RealCheckpoint_RendersSkeletonTensor()
    {
        string? ckpt = Environment.GetEnvironmentVariable("HARTSY_POSE_CKPT");
        if (string.IsNullOrEmpty(ckpt) || !File.Exists(ckpt))
        {
            _output.WriteLine($"SKIPPED: HARTSY_POSE_CKPT missing ({ckpt})");
            return;
        }

        const int width = 512, height = 640;
        byte[] rgb = DrawHumanoid(width, height);

        using IBackend backend = new CpuBackend();
        using OpenPosePreprocessor preprocessor = new(backend, ckpt, inputSize: 640);

        IReadOnlyList<PoseDetection> people = preprocessor.Pipeline.Detect(rgb, width, height, confidenceThreshold: 0.1f);
        _output.WriteLine($"persons detected: {people.Count}");
        foreach (PoseDetection p in people)
            _output.WriteLine($"  conf={p.Confidence:F3} box=({p.Box.X1:F0},{p.Box.Y1:F0})→({p.Box.X2:F0},{p.Box.Y2:F0})");

        using Tensor cond = preprocessor.Process(rgb, width, height, outputWidth: 768, outputHeight: 960,
            confidenceThreshold: 0.1f);
        Assert.Equal(new TensorShape(1, 3, 960, 768), cond.Shape);
        Assert.Equal(DType.F32, cond.DType);

        float* ptr = (float*)cond.DataPointer;
        double sum = 0;
        for (long i = 0; i < cond.Shape.ElementCount; i++)
        {
            Assert.InRange(ptr[i], 0f, 1f);
            sum += ptr[i];
        }
        _output.WriteLine($"skeleton pixel sum: {sum:F1}");

        Assert.NotEmpty(people);
        Assert.True(sum > 0, "detected a person but rendered an empty skeleton");
    }

    /// <summary>Draws a photo-ish standing figure (skin-tone head/hands, shirt, pants, soft shading and noise)
    /// on a textured background — enough person-shape for the nano pose model to latch onto without shipping a
    /// test photo. Proportions follow the ~7.5-heads figure convention.</summary>
    private static byte[] DrawHumanoid(int w, int h)
    {
        byte[] rgb = new byte[w * h * 3];
        Random rng = new Random(1234);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int p = (y * w + x) * 3;
                int shade = 190 + (int)(30f * y / h) + rng.Next(-6, 7);
                rgb[p] = (byte)Math.Clamp(shade, 0, 255);
                rgb[p + 1] = (byte)Math.Clamp(shade + 4, 0, 255);
                rgb[p + 2] = (byte)Math.Clamp(shade + 8, 0, 255);
            }

        byte[] skin = [224, 172, 138], shirt = [60, 90, 160], pants = [50, 50, 60], dark = [40, 30, 25];
        float cx = w * 0.5f;
        float headR = h * 0.045f;

        FillEllipse(rgb, w, h, cx, h * 0.10f, headR * 0.85f, headR, skin);                        // head
        FillEllipse(rgb, w, h, cx, h * 0.072f, headR * 0.9f, headR * 0.55f, dark);                // hair
        FillCapsule(rgb, w, h, cx, h * 0.135f, cx, h * 0.165f, w * 0.022f, skin);                 // neck
        FillEllipse(rgb, w, h, cx, h * 0.30f, w * 0.115f, h * 0.145f, shirt);                     // torso
        FillCapsule(rgb, w, h, cx - w * 0.105f, h * 0.185f, cx - w * 0.16f, h * 0.30f, w * 0.028f, shirt);  // L upper arm
        FillCapsule(rgb, w, h, cx + w * 0.105f, h * 0.185f, cx + w * 0.16f, h * 0.30f, w * 0.028f, shirt);  // R upper arm
        FillCapsule(rgb, w, h, cx - w * 0.16f, h * 0.30f, cx - w * 0.19f, h * 0.42f, w * 0.024f, skin);     // L forearm
        FillCapsule(rgb, w, h, cx + w * 0.16f, h * 0.30f, cx + w * 0.19f, h * 0.42f, w * 0.024f, skin);     // R forearm
        FillEllipse(rgb, w, h, cx - w * 0.19f, h * 0.44f, w * 0.026f, h * 0.02f, skin);                     // L hand
        FillEllipse(rgb, w, h, cx + w * 0.19f, h * 0.44f, w * 0.026f, h * 0.02f, skin);                     // R hand
        FillCapsule(rgb, w, h, cx - w * 0.055f, h * 0.44f, cx - w * 0.07f, h * 0.65f, w * 0.038f, pants);   // L thigh
        FillCapsule(rgb, w, h, cx + w * 0.055f, h * 0.44f, cx + w * 0.07f, h * 0.65f, w * 0.038f, pants);   // R thigh
        FillCapsule(rgb, w, h, cx - w * 0.07f, h * 0.65f, cx - w * 0.075f, h * 0.87f, w * 0.03f, pants);    // L shin
        FillCapsule(rgb, w, h, cx + w * 0.07f, h * 0.65f, cx + w * 0.075f, h * 0.87f, w * 0.03f, pants);    // R shin
        FillEllipse(rgb, w, h, cx - w * 0.09f, h * 0.895f, w * 0.045f, h * 0.017f, dark);                   // L shoe
        FillEllipse(rgb, w, h, cx + w * 0.09f, h * 0.895f, w * 0.045f, h * 0.017f, dark);                   // R shoe

        // Face features so the head reads as a face, not a blob.
        FillEllipse(rgb, w, h, cx - headR * 0.35f, h * 0.097f, w * 0.008f, h * 0.005f, dark);     // L eye
        FillEllipse(rgb, w, h, cx + headR * 0.35f, h * 0.097f, w * 0.008f, h * 0.005f, dark);     // R eye
        FillEllipse(rgb, w, h, cx, h * 0.122f, w * 0.014f, h * 0.004f, [150, 90, 80]);            // mouth
        return rgb;
    }

    private static void FillEllipse(byte[] rgb, int w, int h, float cx, float cy, float rx, float ry, byte[] color)
    {
        int ix0 = Math.Max(0, (int)(cx - rx)), ix1 = Math.Min(w - 1, (int)(cx + rx));
        int iy0 = Math.Max(0, (int)(cy - ry)), iy1 = Math.Min(h - 1, (int)(cy + ry));
        for (int y = iy0; y <= iy1; y++)
            for (int x = ix0; x <= ix1; x++)
            {
                float dx = (x - cx) / rx, dy = (y - cy) / ry;
                float e = dx * dx + dy * dy;
                if (e > 1f) continue;
                ShadePixel(rgb, w, x, y, color, e);
            }
    }

    /// <summary>Thick line segment with round caps — limbs read better than axis-aligned boxes.</summary>
    private static void FillCapsule(byte[] rgb, int w, int h, float x0, float y0, float x1, float y1, float r, byte[] color)
    {
        float dx = x1 - x0, dy = y1 - y0;
        float len2 = dx * dx + dy * dy;
        int ix0 = Math.Max(0, (int)(MathF.Min(x0, x1) - r)), ix1 = Math.Min(w - 1, (int)(MathF.Max(x0, x1) + r));
        int iy0 = Math.Max(0, (int)(MathF.Min(y0, y1) - r)), iy1 = Math.Min(h - 1, (int)(MathF.Max(y0, y1) + r));
        for (int y = iy0; y <= iy1; y++)
            for (int x = ix0; x <= ix1; x++)
            {
                float t = len2 > 0 ? Math.Clamp(((x - x0) * dx + (y - y0) * dy) / len2, 0f, 1f) : 0f;
                float px = x0 + t * dx, py = y0 + t * dy;
                float ex = x - px, ey = y - py;
                float d2 = (ex * ex + ey * ey) / (r * r);
                if (d2 > 1f) continue;
                ShadePixel(rgb, w, x, y, color, d2);
            }
    }

    /// <summary>Writes a color with edge darkening + light noise so shapes look shaded rather than flat-filled.</summary>
    private static void ShadePixel(byte[] rgb, int w, int x, int y, byte[] color, float edge)
    {
        float shade = 1f - 0.35f * edge;
        int jitter = ((x * 31 + y * 17) % 7) - 3;
        int p = (y * w + x) * 3;
        rgb[p] = (byte)Math.Clamp((int)(color[0] * shade) + jitter, 0, 255);
        rgb[p + 1] = (byte)Math.Clamp((int)(color[1] * shade) + jitter, 0, 255);
        rgb[p + 2] = (byte)Math.Clamp((int)(color[2] * shade) + jitter, 0, 255);
    }
}
