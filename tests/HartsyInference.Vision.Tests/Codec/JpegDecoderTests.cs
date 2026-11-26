using HartsyInference.Vision.Codec;
using Xunit;

namespace HartsyInference.Vision.Tests.Codec;

/// <summary>The baseline JPEG decoder against libjpeg. The fixtures are small files written by Pillow; the expected pixels are Pillow's decodes at the
/// sampled coordinates. Chroma is upsampled by nearest neighbour here and triangle-filtered by libjpeg, so colour samples are held to a few levels.</summary>
public sealed class JpegDecoderTests
{
    // 32×24, 4:2:0, quality 95, a restart marker every 2 MCUs (Pillow restart_marker_blocks=2).
    private const string ColourRestart =
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAAYACADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/90ABAAC/9oADAMBAAIRAxEAPwD8r/BvwY+5/on/AI7Xq/g34Mfc/wBE/wDHa9d8G/Bj7n+if+O16x4N+DH3P9E/8dr+g4eJ/wDf/E+N8OvEn4Pf7HkXg34Mfc/0T/x2vV/BvwY+5/on/jteu+Dfgx9z/RP/AB2vWPBvwY+7/on/AI7XVDxP/vn94eHXiT8Hvn//0Nnwb8GPuH7J/wCO16v4N+DH3P8ARP8Ax2iivxmGf5n/ADn+evh1nuZe57/Y9Y8G/Bj7n+if+O16v4N+DH3P9E/8doorqhxBmf8AOf3h4dZ7mXue/wBj/9k=";

    // 16×16 grayscale, quality 95.
    private const string Grey =
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBgYFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/wAALCAAQABABAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APgn/gm34I/48P3P93tX7wfsaa34E+Bvwr1b41/FLV/7L8MeD/D91rfiPU/sks/2SwtIHnuJvLhV5JNkUbttRWZsYUEkCvx//wCCbfgj/jw/c/3e1fph+35418VfAH/gi18cfHXg7StPubu/8HweG5o9Thd4xa6zfWukXTgI6HzVt76V4ySVEioWV1BRv//Z";

    // 64×48 progressive, quality 90 (Pillow progressive=True): not a baseline file.
    private const string Progressive =
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wgARCAAYACADASIAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAAAAAABQAH/8QAGQEAAQUAAAAAAAAAAAAAAAAABwIDBQYI/9oADAMBAAIQAxAAAAHD3X3bXHAOvuuFAR2hdmd12llD/8QAFhAAAwAAAAAAAAAAAAAAAAAAAAME/9oACAEBAAEFAkxiYxMYmMTGJjExiYhMYmMTGJjP/8QAGREAAgMBAAAAAAAAAAAAAAAAAAUCAyEx/9oACAEDAQE/AVzLmi5lzRdfPBdfPD//xAAbEQABBAMAAAAAAAAAAAAAAAAAAxVRYQEEFP/aAAgBAgEBPwFzsc7OhSTGwpJ//8QAFRABAQAAAAAAAAAAAAAAAAAAADH/2gAIAQEABj8CiIiIiIiI/8QAFRABAQAAAAAAAAAAAAAAAAAAADH/2gAIAQEAAT8hiRIkSJEiQIkSJE//2gAMAwEAAgADAAAAEA3ff//EABYRAQEBAAAAAAAAAAAAAAAAAABBEP/aAAgBAwEBPxDASsrP/8QAFhEBAQEAAAAAAAAAAAAAAAAAAQAQ/9oACAECAQE/ECijAH//xAAVEAEBAAAAAAAAAAAAAAAAAAAA8f/aAAgBAQABPxCYmJiYmJiYkJiYmJj/2Q==";

    private static byte[] Bytes(string b64) => Convert.FromBase64String(b64);

    private static (int R, int G, int B) At((byte[] Rgb, int Width, int Height) img, int x, int y)
    {
        int o = (y * img.Width + x) * 3;
        return (img.Rgb[o], img.Rgb[o + 1], img.Rgb[o + 2]);
    }

    [Fact]
    public void TheSoiMarkerIdentifiesAJpeg()
    {
        Assert.True(JpegDecoder.IsJpeg(Bytes(Grey)));
        Assert.False(JpegDecoder.IsJpeg(new byte[] { 0x89, 0x50, 0x4E, 0x47 }));
    }

    [Fact]
    public void AColourBaselineFileWithRestartMarkersMatchesLibjpegWithinAFewLevels()
    {
        (byte[] Rgb, int Width, int Height) img = JpegDecoder.Decode(Bytes(ColourRestart));
        Assert.Equal((32, 24), (img.Width, img.Height));
        // Pillow's decodes at the sampled coordinates; nearest-neighbour chroma leaves a few levels of difference.
        (int x, int y, int r, int g, int b)[] want =
        [
            (0, 0, 0, 2, 125), (31, 0, 241, 4, 216), (0, 23, 5, 226, 87),
            (31, 23, 250, 228, 181), (15, 11, 120, 109, 151), (7, 19, 55, 190, 111),
        ];
        foreach ((int x, int y, int r, int g, int b) in want)
        {
            (int R, int G, int B) p = At(img, x, y);
            int worst = Math.Max(Math.Abs(p.R - r), Math.Max(Math.Abs(p.G - g), Math.Abs(p.B - b)));
            Assert.True(worst <= 12, $"pixel ({x},{y}): got {p}, libjpeg ({r},{g},{b}), worst channel {worst}");
        }
    }

    [Fact]
    public void AGreyscaleFileDecodesToEqualChannels()
    {
        (byte[] Rgb, int Width, int Height) img = JpegDecoder.Decode(Bytes(Grey));
        Assert.Equal((16, 16), (img.Width, img.Height));
        foreach ((int x, int y, int want) in new[] { (0, 0, 0), (15, 15, 29), (7, 3, 121) })
        {
            (int R, int G, int B) p = At(img, x, y);
            Assert.True(p.R == p.G && p.G == p.B, $"grey pixel at ({x},{y}) is not grey: {p}");
            Assert.True(Math.Abs(p.R - want) <= 3, $"grey ({x},{y}): {p.R} vs libjpeg {want}");
        }
    }

    [Fact]
    public void AProgressiveFileIsRefusedWithTheReason()
    {
        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => JpegDecoder.Decode(Bytes(Progressive)));
        Assert.Contains("progressive", ex.Message);
    }

    [Fact]
    public void BytesThatAreNotAJpegAreRefused()
    {
        Assert.Throws<ArgumentException>(() => JpegDecoder.Decode(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0, 0, 0, 0 }));
    }

    private static int Marker(byte[] d, byte m)
    {
        for (int i = 0; i + 1 < d.Length; i++) if (d[i] == 0xFF && d[i + 1] == m) return i;
        throw new InvalidOperationException($"marker FF{m:X2} not in the fixture");
    }

    [Fact]
    public void AFrameLargerThanTheDecoderLimitIsRefusedBeforeAnyAllocation()
    {
        byte[] d = Bytes(Grey);
        int sof = Marker(d, 0xC0);
        d[sof + 5] = 0xFF; d[sof + 6] = 0xFF;   // height 65535
        d[sof + 7] = 0xFF; d[sof + 8] = 0xFF;   // width 65535
        ArgumentException ex = Assert.Throws<ArgumentException>(() => JpegDecoder.Decode(d));
        Assert.Contains("limit", ex.Message);
    }

    [Fact]
    public void ATableSelectorAboveThreeIsRefusedNotIndexed()
    {
        byte[] d = Bytes(Grey);
        int sos = Marker(d, 0xDA);
        d[sos + 6] = 0x40;   // DC table 4, one past the four-entry table set
        ArgumentException ex = Assert.Throws<ArgumentException>(() => JpegDecoder.Decode(d));
        Assert.Contains("Huffman table", ex.Message);
    }

    [Fact]
    public void ATruncatedFileIsRefusedAsAnArgumentNotAnIndexFault()
    {
        byte[] d = Bytes(ColourRestart)[..120];
        ArgumentException ex = Assert.Throws<ArgumentException>(() => JpegDecoder.Decode(d));
        Assert.Contains("Not a decodable JPEG", ex.Message);
    }

    [Fact]
    public void AScanThatEndsBeforeItsLastMcuIsRefusedNotFilledWithZeros()
    {
        // Cut the scan short inside its entropy data, then append EOI: the bit reader meets the marker before the last MCU.
        byte[] full = Bytes(Grey);
        int sos = Marker(full, 0xDA);
        int eoi = Marker(full, 0xD9);
        byte[] cut = new byte[sos + 12 + 4];
        Array.Copy(full, cut, sos + 12);
        cut[^4] = 0xFF; cut[^3] = 0xD9;
        Assert.Throws<ArgumentException>(() => JpegDecoder.Decode(cut));
        Assert.True(eoi > sos);
    }
}
