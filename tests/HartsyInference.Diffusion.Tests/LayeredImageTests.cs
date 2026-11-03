using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The layered-output rules: the composite is straight-alpha "over" of the layers, resize works on premultiplied
/// values so transparent colour never bleeds, crop cuts every layer alike, and the post-steps with no layered rule are
/// refused.</summary>
public sealed class LayeredImageTests
{
    [Fact]
    public void Composite_IsOverBottomFirstWithStraightAlpha()
    {
        // Bottom: opaque blue. Top: red at half coverage.
        ImageLayer bottom = Solid(0, 2, 1, 0, 0, 255, 255);
        ImageLayer top = Solid(1, 2, 1, 255, 0, 0, 128);

        (byte[] rgb, byte[] alpha) = LayeredImage.Composite([bottom, top]);

        Assert.Equal(255, alpha[0]);
        // 128/255 red over blue: R = 255·0.502, B = 255·0.498.
        Assert.Equal(new byte[] { 128, 0, 127 }, rgb[..3]);
    }

    [Fact]
    public void Composite_OfTranslucentLayersKeepsStraightColour()
    {
        ImageLayer a = Solid(0, 1, 1, 200, 100, 50, 100);
        ImageLayer b = Solid(1, 1, 1, 200, 100, 50, 100);

        (byte[] rgb, byte[] alpha) = LayeredImage.Composite([a, b]);

        // Same colour over itself stays that colour; coverage grows to 1 − (1 − 0.392)².
        Assert.Equal(new byte[] { 200, 100, 50 }, rgb);
        Assert.Equal(161, alpha[0]);
    }

    [Fact]
    public void Composite_FullyTransparentPixelIsBlackAndClear()
    {
        (byte[] rgb, byte[] alpha) = LayeredImage.Composite([Solid(0, 1, 1, 255, 255, 255, 0)]);

        Assert.Equal(new byte[] { 0, 0, 0 }, rgb);
        Assert.Equal(0, alpha[0]);
    }

    [Fact]
    public void Resize_DoesNotBleedTransparentColour()
    {
        // One layer: an opaque blue pixel next to a fully transparent pure-red pixel. Downscaling 2→1 must give blue at
        // half coverage, with no red; resampling straight colour would give purple.
        ImageLayer layer = new ImageLayer
        {
            Index = 0, Width = 2, Height = 1,
            Rgb = [0, 0, 255, 255, 0, 0],
            Alpha = [255, 0],
        };
        ImageResult result = Result(2, 1, layer);

        ImageResult small = LayeredImage.Resize(result, 1, 1);

        ImageLayer resized = Assert.Single(small.Layers!);
        Assert.Equal(new byte[] { 0, 0, 255 }, resized.Rgb);
        Assert.Equal(128, resized.Alpha[0]);
        Assert.Equal(resized.Rgb, small.Rgb);
        Assert.Equal(resized.Alpha, small.Alpha);
    }

    [Fact]
    public void Resize_UpscaleOfAUniformLayerStaysUniform()
    {
        ImageResult result = Result(2, 2, Solid(0, 2, 2, 10, 20, 30, 200));

        ImageResult big = LayeredImage.Resize(result, 5, 3);

        Assert.Equal((5, 3), (big.Width, big.Height));
        Assert.All(Enumerable.Range(0, 15), p => Assert.Equal(new byte[] { 10, 20, 30 }, big.Layers![0].Rgb[(p * 3)..(p * 3 + 3)]));
        Assert.All(big.Layers![0].Alpha, a => Assert.Equal(200, a));
    }

    [Fact]
    public void Crop_CutsEveryLayerAndRecomposites()
    {
        ImageLayer bottom = new ImageLayer
        {
            Index = 0, Width = 3, Height = 1,
            Rgb = [1, 1, 1, 2, 2, 2, 3, 3, 3],
            Alpha = [255, 255, 255],
        };
        ImageLayer top = new ImageLayer
        {
            Index = 1, Width = 3, Height = 1,
            Rgb = [9, 9, 9, 8, 8, 8, 7, 7, 7],
            Alpha = [0, 255, 0],
        };

        ImageResult cropped = LayeredImage.Crop(Result(3, 1, bottom, top), 1, 0, 2, 1);

        Assert.Equal(new byte[] { 2, 2, 2, 3, 3, 3 }, cropped.Layers![0].Rgb);
        Assert.Equal(new byte[] { 255, 0 }, cropped.Layers[1].Alpha);
        Assert.Equal(new byte[] { 8, 8, 8, 3, 3, 3 }, cropped.Rgb);
    }

    [Fact]
    public void Validate_RejectsOutOfOrderOrMissizedLayers()
    {
        Assert.Throws<InvalidOperationException>(() => LayeredImage.Validate(Result(1, 1, Solid(1, 1, 1, 0, 0, 0, 255))));
        Assert.Throws<InvalidOperationException>(() => LayeredImage.Validate(Result(2, 1, Solid(0, 1, 1, 0, 0, 0, 255))));
        Assert.Throws<InvalidOperationException>(() => LayeredImage.Validate(new ImageResult { Rgb = [0, 0, 0], Width = 1, Height = 1, Layers = [] }));
    }

    [Fact]
    public void Violation_NamesEachPostStepWithNoLayeredRule()
    {
        ImageRequest plain = new ImageRequest { Prompt = "a cat" };
        Assert.Null(LayeredImage.Violation(plain));
        Assert.Contains("Inpainting", LayeredImage.Violation(plain with
        {
            Inpaint = new Inpaint { Mask = new ImageData { Rgb = new byte[3], Width = 1, Height = 1 } },
            Img2Img = new Img2Img { InitImage = new ImageData { Rgb = new byte[3], Width = 1, Height = 1 } },
        }));
        Assert.Contains("refiner", LayeredImage.Violation(plain with { Refiner = new Refiner { Model = "x" } }));
        Assert.Contains("Background removal", LayeredImage.Violation(plain with { RemoveBackground = true }));
        Assert.Contains("Segment refinement", LayeredImage.Violation(plain with { Prompt = "a cat <segment:face> smiling" }));
    }

    private static ImageLayer Solid(int index, int width, int height, byte r, byte g, byte b, byte a)
    {
        int pixels = width * height;
        byte[] rgb = new byte[pixels * 3];
        for (int p = 0; p < pixels; p++)
        {
            rgb[p * 3] = r;
            rgb[p * 3 + 1] = g;
            rgb[p * 3 + 2] = b;
        }
        return new ImageLayer { Index = index, Width = width, Height = height, Rgb = rgb, Alpha = Enumerable.Repeat(a, pixels).ToArray() };
    }

    private static ImageResult Result(int width, int height, params ImageLayer[] layers) =>
        new ImageResult { Rgb = new byte[width * height * 3], Width = width, Height = height, Layers = layers };
}
