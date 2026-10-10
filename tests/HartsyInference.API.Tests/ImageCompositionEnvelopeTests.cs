using HartsyInference.API.Endpoints;
using HartsyInference.Engine;
using HartsyInference.Engine.Requests;
using EngineImageData = HartsyInference.Engine.Requests.ImageData;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>The <c>/v1/native/images</c> envelope's img2img conveniences. The native contract carries raw RGB24, which
/// an HTTP client never has — these fields let a caller send an ordinary PNG, so they are the difference between
/// img2img being reachable over HTTP and being theoretically expressible.</summary>
public sealed class ImageCompositionEnvelopeTests
{
    private static string PngBase64(int width, int height, byte value)
    {
        byte[] rgb = Enumerable.Repeat(value, width * height * 3).ToArray();
        return Convert.ToBase64String(PngEncoder.Encode(rgb, width, height));
    }

    private static NativeImageRequest Envelope() => new NativeImageRequest
    {
        Model = "sdxl",
        Request = new ImageRequest { Prompt = "test" },
    };

    [Fact]
    public void InitImageBase64_DecodesIntoTheImg2ImgComposition()
    {
        NativeImageRequest req = Envelope();
        req.InitImageBase64 = PngBase64(24, 16, 128);
        req.Creativity = 0.35;

        ImageRequest result = ImageEndpoints.ApplyImageComposition(req);

        Assert.NotNull(result.Img2Img);
        Assert.Equal(24, result.Img2Img!.InitImage.Width);
        Assert.Equal(16, result.Img2Img.InitImage.Height);
        Assert.Equal(24 * 16 * 3, result.Img2Img.InitImage.Rgb.Length);
        Assert.Equal(0.35, result.Img2Img.Creativity);
    }

    [Fact]
    public void MaskBase64_DecodesWithItsGrowBlurAndShrinkGrowKnobs()
    {
        NativeImageRequest req = Envelope();
        req.InitImageBase64 = PngBase64(32, 32, 90);
        req.MaskBase64 = PngBase64(32, 32, 255);
        req.MaskGrow = 4;
        req.MaskBlur = 6;
        req.MaskShrinkGrow = 12;

        ImageRequest result = ImageEndpoints.ApplyImageComposition(req);

        Assert.NotNull(result.Inpaint);
        Assert.Equal(32, result.Inpaint!.Mask.Width);
        Assert.Equal(4, result.Inpaint.Grow);
        Assert.Equal(6, result.Inpaint.Blur);
        Assert.Equal(12, result.Inpaint.ShrinkGrow);
    }

    /// <summary>Creativity alone must not invent an init image, but should retune one the caller already supplied
    /// through the raw contract.</summary>
    [Fact]
    public void Creativity_AloneRetunesAnExistingInitImageAndNeverCreatesOne()
    {
        NativeImageRequest without = Envelope();
        without.Creativity = 0.9;
        Assert.Null(ImageEndpoints.ApplyImageComposition(without).Img2Img);

        NativeImageRequest with = new NativeImageRequest
        {
            Model = "sdxl",
            Request = new ImageRequest
            {
                Prompt = "test",
                Img2Img = new Img2Img
                {
                    InitImage = new EngineImageData { Rgb = new byte[8 * 8 * 3], Width = 8, Height = 8 },
                    Creativity = 0.6,
                },
            },
            Creativity = 0.9,
        };
        Assert.Equal(0.9, ImageEndpoints.ApplyImageComposition(with).Img2Img!.Creativity);
    }

    [Fact]
    public void ANonPngPayload_FailsNamingTheFormat()
    {
        NativeImageRequest req = Envelope();
        // JPEG signature.
        req.InitImageBase64 = Convert.ToBase64String([0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0, 0, 0]);

        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => ImageEndpoints.ApplyImageComposition(req));
        Assert.Contains("JPEG", ex.Message, StringComparison.Ordinal);
    }
}
