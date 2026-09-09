using System.Buffers.Binary;
using System.IO.Compression;
using HartsyInference.Engine;
using EngineImageData = HartsyInference.Engine.Requests.ImageData;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>RGBA output of the in-house PNG encoder: the IHDR colour type and the interleaved scanline bytes are
/// what a downstream client (HartsyWeb's cutout tool inspects IHDR for colour type 6) keys on.</summary>
public sealed class PngEncoderAlphaTests
{
    [Fact]
    public void Encode_WithAlpha_WritesColorType6AndInterleavesAlpha()
    {
        EngineImageData image = new()
        {
            Rgb = [10, 20, 30, 40, 50, 60],
            Alpha = [255, 0],
            Width = 2,
            Height = 1,
        };

        byte[] png = PngEncoder.Encode(image);

        // Signature (8) + IHDR length/type (8) + width/height (8): bit depth at 24, colour type at 25.
        Assert.Equal(8, png[24]);
        Assert.Equal(6, png[25]);
        byte[] scanlines = InflateIdat(png);
        Assert.Equal(new byte[] { 0, 10, 20, 30, 255, 40, 50, 60, 0 }, scanlines);
    }

    [Fact]
    public void Encode_WithoutAlpha_StaysColorType2()
    {
        EngineImageData image = new() { Rgb = [1, 2, 3], Width = 1, Height = 1 };
        byte[] png = PngEncoder.Encode(image);
        Assert.Equal(2, png[25]);
        Assert.Equal(new byte[] { 0, 1, 2, 3 }, InflateIdat(png));
    }

    [Fact]
    public void Encode_IgnoresMissizedAlpha()
    {
        // A plane that does not cover the image is not an alpha channel; the record's HasAlpha says so.
        EngineImageData image = new() { Rgb = [1, 2, 3], Alpha = [255, 255], Width = 1, Height = 1 };
        Assert.False(image.HasAlpha);
        Assert.Equal(2, PngEncoder.Encode(image)[25]);
    }

    private static byte[] InflateIdat(byte[] png)
    {
        int offset = 8;
        while (offset < png.Length)
        {
            int length = BinaryPrimitives.ReadInt32BigEndian(png.AsSpan(offset, 4));
            string type = System.Text.Encoding.ASCII.GetString(png, offset + 4, 4);
            if (type == "IDAT")
            {
                using MemoryStream compressed = new(png, offset + 8, length);
                using ZLibStream zlib = new(compressed, CompressionMode.Decompress);
                using MemoryStream raw = new();
                zlib.CopyTo(raw);
                return raw.ToArray();
            }
            offset += 12 + length;
        }
        throw new InvalidOperationException("No IDAT chunk.");
    }
}
