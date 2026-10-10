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
}
