using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Vision.Tests.Codec;

public sealed class PngAlphaAndOrientationTests
{
    [Fact]
    public void DecodeWithAlpha_ReturnsTheAlphaPlaneOfAnRgbaFile()
    {
        byte[] png = Png(width: 2, height: 1, colorType: 6, pixels: [1, 2, 3, 200, 4, 5, 6, 7]);

        DecodedPng decoded = PngDecoder.DecodeWithAlpha(png);

        Assert.Equal([1, 2, 3, 4, 5, 6], decoded.Rgb);
        Assert.Equal([200, 7], decoded.Alpha);
        Assert.Equal(1, decoded.ExifOrientation);
        Assert.False(decoded.OrientationApplied);
    }

    [Fact]
    public void DecodeWithAlpha_TurnsAColourKeyIntoTransparency()
    {
        // tRNS of an RGB file: the 16-bit key (10, 20, 30); only the first pixel matches it
        byte[] trns = [0, 10, 0, 20, 0, 30];
        byte[] png = Png(width: 2, height: 1, colorType: 2, pixels: [10, 20, 30, 40, 50, 60], Chunk("tRNS", trns));

        DecodedPng decoded = PngDecoder.DecodeWithAlpha(png);

        Assert.Equal([0, 255], decoded.Alpha);
    }

    [Fact]
    public void DecodeWithAlpha_HasNoAlphaPlaneForAPlainRgbFile()
    {
        byte[] png = Png(width: 1, height: 1, colorType: 2, pixels: [1, 2, 3]);

        DecodedPng decoded = PngDecoder.DecodeWithAlpha(png);

        Assert.Null(decoded.Alpha);
    }

    [Fact]
    public void DecodeWithAlpha_AppliesTheExifOrientationAsPilDoes()
    {
        // 2 x 3 image; the red channel holds the pixel index (row-major): orientation 6 turns it clockwise into 3 x 2
        byte[] pixels = new byte[2 * 3 * 3];
        for (int i = 0; i < 6; i++)
        {
            pixels[i * 3] = (byte)i;
        }

        byte[] png = Png(width: 2, height: 3, colorType: 2, pixels: pixels, Chunk("eXIf", Tiff(6)));

        DecodedPng decoded = PngDecoder.DecodeWithAlpha(png);

        Assert.Equal(3, decoded.Width);
        Assert.Equal(2, decoded.Height);
        Assert.Equal([4, 2, 0, 5, 3, 1], Reds(decoded.Rgb));
        Assert.Equal(6, decoded.ExifOrientation);
        Assert.True(decoded.OrientationApplied);
    }

    [Fact]
    public void Decode_LeavesTheOrientationAsStored()
    {
        byte[] pixels = new byte[2 * 3 * 3];
        for (int i = 0; i < 6; i++)
        {
            pixels[i * 3] = (byte)i;
        }

        byte[] png = Png(width: 2, height: 3, colorType: 2, pixels: pixels, Chunk("eXIf", Tiff(6)));

        (byte[] rgb, int width, int height) = PngDecoder.Decode(png);

        Assert.Equal(2, width);
        Assert.Equal(3, height);
        Assert.Equal([0, 1, 2, 3, 4, 5], Reds(rgb));
    }

    [Fact]
    public void ReadTiff_ReturnsTheOrientationOrUprightWhenMalformed()
    {
        Assert.Equal(8, ExifOrientation.ReadTiff(Tiff(8)));
        Assert.Equal(1, ExifOrientation.ReadTiff(null));
        Assert.Equal(1, ExifOrientation.ReadTiff([1, 2, 3]));
        Assert.Equal(1, ExifOrientation.ReadTiff(Tiff(9)));
    }

    [Fact]
    public void Apply_IsTheIdentityForUpright()
    {
        byte[] rgb = [1, 2, 3];

        (byte[] outRgb, byte[]? alpha, int width, int height) = ExifOrientation.Apply(rgb, null, 1, 1, ExifOrientation.Upright);

        Assert.Same(rgb, outRgb);
        Assert.Null(alpha);
        Assert.Equal((1, 1), (width, height));
    }

    [Fact]
    public void Apply_RejectsAnOrientationOutsideOneToEight()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ExifOrientation.Apply([1, 2, 3], null, 1, 1, 9));
    }

    private static int[] Reds(byte[] rgb)
    {
        int[] reds = new int[rgb.Length / 3];
        for (int i = 0; i < reds.Length; i++)
        {
            reds[i] = rgb[i * 3];
        }

        return reds;
    }

    /// <summary>A minimal PNG with 8-bit samples and filter 0; the CRCs are not checked by the decoder, so they are zero.</summary>
    private static byte[] Png(int width, int height, int colorType, byte[] pixels, params byte[][] extraChunks)
    {
        int channels = colorType == 6 ? 4 : 3;
        int stride = width * channels;
        byte[] filtered = new byte[height * (stride + 1)];
        for (int y = 0; y < height; y++)
        {
            Array.Copy(pixels, y * stride, filtered, y * (stride + 1) + 1, stride);
        }

        byte[] ihdr = new byte[13];
        BinaryPrimitives.WriteInt32BigEndian(ihdr.AsSpan(0), width);
        BinaryPrimitives.WriteInt32BigEndian(ihdr.AsSpan(4), height);
        ihdr[8] = 8;
        ihdr[9] = (byte)colorType;

        using MemoryStream compressed = new();
        using (ZLibStream zlib = new(compressed, CompressionLevel.Optimal, leaveOpen: true))
        {
            zlib.Write(filtered);
        }

        List<byte> png = [137, 80, 78, 71, 13, 10, 26, 10];
        png.AddRange(Chunk("IHDR", ihdr));
        foreach (byte[] chunk in extraChunks)
        {
            png.AddRange(chunk);
        }

        png.AddRange(Chunk("IDAT", compressed.ToArray()));
        png.AddRange(Chunk("IEND", []));
        return [.. png];
    }

    private static byte[] Chunk(string type, byte[] body)
    {
        byte[] chunk = new byte[12 + body.Length];
        BinaryPrimitives.WriteInt32BigEndian(chunk.AsSpan(0), body.Length);
        Encoding.ASCII.GetBytes(type).CopyTo(chunk, 4);
        body.CopyTo(chunk, 8);
        return chunk;
    }

    /// <summary>A little-endian TIFF block whose only IFD entry is the orientation tag.</summary>
    private static byte[] Tiff(int orientation)
    {
        byte[] tiff = new byte[8 + 2 + 12 + 4];
        tiff[0] = (byte)'I';
        tiff[1] = (byte)'I';
        BinaryPrimitives.WriteUInt16LittleEndian(tiff.AsSpan(2), 42);
        BinaryPrimitives.WriteUInt32LittleEndian(tiff.AsSpan(4), 8);
        BinaryPrimitives.WriteUInt16LittleEndian(tiff.AsSpan(8), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(tiff.AsSpan(10), 0x0112);
        BinaryPrimitives.WriteUInt16LittleEndian(tiff.AsSpan(12), 3);
        BinaryPrimitives.WriteUInt32LittleEndian(tiff.AsSpan(14), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(tiff.AsSpan(18), (ushort)orientation);
        return tiff;
    }
}
