using HartsyInference.Vision.Codec;
using Xunit;

namespace HartsyInference.Vision.Tests;

/// <summary>Tests for the pure-C# PNG decoder. Uses a bundled test fixture (<c>TestData/bus.png</c>)
/// as the real-image input — a 810×1080 RGB PNG from Ultralytics' standard test set.</summary>
public sealed class PngDecoderTests
{
    private static string TestImagePath
    {
        get
        {
            string baseDir = AppContext.BaseDirectory;
            return Path.Combine(baseDir, "TestData", "bus.png");
        }
    }

    [Fact]
    public void DecodeFromFile_LoadsBusPng_WithExpectedDimensions()
    {
        Assert.True(File.Exists(TestImagePath), $"TestData/bus.png missing at {TestImagePath} — should be copied to output dir.");
        (byte[] rgb, int width, int height) = PngDecoder.DecodeFromFile(TestImagePath);
        Assert.Equal(810, width);
        Assert.Equal(1080, height);
        Assert.Equal(810 * 1080 * 3, rgb.Length);
    }

    [Fact]
    public void Decode_RejectsNonPngBytes()
    {
        byte[] notPng = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46]; // JPEG SOI
        Assert.Throws<ArgumentException>(() => PngDecoder.Decode(notPng));
    }

    [Fact]
    public void Decode_ProducesPlausiblePixelValues()
    {
        (byte[] rgb, _, _) = PngDecoder.DecodeFromFile(TestImagePath);
        // The bus.jpg image is a photograph — pixel values should span the full 0..255 range with
        // non-trivial variance. A broken decoder typically produces all-zero, constant, or wildly
        // oscillating data — this catches all three.
        int minR = 255, maxR = 0;
        long sumR = 0;
        int stride = 3;
        int sampleCount = 10000; // sample a subset for speed
        int step = rgb.Length / (stride * sampleCount);
        if (step < 1) step = 1;
        int sampled = 0;
        for (int i = 0; i < rgb.Length; i += stride * step)
        {
            byte r = rgb[i];
            if (r < minR) minR = r;
            if (r > maxR) maxR = r;
            sumR += r;
            sampled++;
        }
        Assert.True(maxR - minR > 100, $"Pixel range too narrow: min={minR}, max={maxR} — decoder may be producing constant/zero data.");
        double meanR = sumR / (double)sampled;
        Assert.InRange(meanR, 30, 230);
    }
}
