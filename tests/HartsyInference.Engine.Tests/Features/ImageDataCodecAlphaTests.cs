using HartsyInference.Engine;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Engine.Tests.Features;

/// <summary>The base64 PNG input of the mesh route keeps its alpha, so HairDeck's matte survives the HTTP path.</summary>
public sealed class ImageDataCodecAlphaTests
{
    [Fact]
    public void DecodeBase64WithAlpha_KeepsTheAlphaPlane()
    {
        byte[] rgb = [10, 20, 30, 40, 50, 60];
        byte[] alpha = [0, 255];
        string png = Convert.ToBase64String(PngEncoder.Encode(rgb, alpha, 2, 1));

        ImageData image = ImageDataCodec.DecodeBase64WithAlpha(png);

        Assert.Equal(2, image.Width);
        Assert.Equal(1, image.Height);
        Assert.True(image.HasAlpha);
        Assert.Equal(alpha, image.Alpha);
        Assert.Equal(rgb, image.Rgb);
    }

    [Fact]
    public void DecodeBase64WithAlpha_AcceptsADataUriPrefix()
    {
        string png = Convert.ToBase64String(PngEncoder.Encode([1, 2, 3], null, 1, 1));

        ImageData image = ImageDataCodec.DecodeBase64WithAlpha("data:image/png;base64," + png);

        Assert.Equal(1, image.Width);
        Assert.False(image.HasAlpha);
    }

    [Fact]
    public void DecodeBase64WithAlpha_RefusesAFormatThatIsNeitherPngNorJpeg()
    {
        // A GIF header: not PNG and not JPEG, so the codec names the format it does not decode (NotSupported; the API answers 501).
        string gif = Convert.ToBase64String([0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00]);   // "GIF89a" plus a 1x1 logical screen

        Assert.Throws<NotSupportedException>(() => ImageDataCodec.DecodeBase64WithAlpha(gif));
    }

    [Fact]
    public void DecodeBase64WithAlpha_AMalformedJpegIsAnArgumentErrorTheApiAnswersAs400()
    {
        // JPEG magic, then a truncated header: the JPEG decoder rejects it as bad input (ArgumentException; the API maps it to 400).
        string jpeg = Convert.ToBase64String([0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0]);

        Assert.Throws<ArgumentException>(() => ImageDataCodec.DecodeBase64WithAlpha(jpeg));
    }
}
