using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Behaviour of <see cref="MaskRecomposite"/> and the hard-threshold policy it shares with
/// <see cref="InpaintOnlyMasked.Composite"/>. Weight-free raster checks.</summary>
public sealed class MaskRecompositeTests
{
    private const int Size = 32;
    private const byte Original = 40;
    private const byte Generated = 200;
    private const byte HalfMask = 100;

    private static ImageData Solid(byte value) =>
        new ImageData { Rgb = Enumerable.Repeat(value, Size * Size * 3).ToArray(), Width = Size, Height = Size };

    /// <summary>Left half at <paramref name="leftValue"/>, right half black.</summary>
    private static ImageData HalfMaskImage(byte leftValue)
    {
        byte[] rgb = new byte[Size * Size * 3];
        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size / 2; x++)
            {
                int idx = (y * Size + x) * 3;
                rgb[idx] = rgb[idx + 1] = rgb[idx + 2] = leftValue;
            }
        }
        return new ImageData { Rgb = rgb, Width = Size, Height = Size };
    }

    private static ImageRequest Request(ImageData mask, bool recomposite = true, bool unthresholded = false) =>
        new ImageRequest
        {
            Prompt = "test",
            Img2Img = new Img2Img { InitImage = Solid(Original) },
            Inpaint = new Inpaint { Mask = mask, RecompositeMask = recomposite },
            MaskCompositeUnthresholded = unthresholded,
        };

    private static ImageResult Result() =>
        new ImageResult
        {
            Rgb = Enumerable.Repeat(Generated, Size * Size * 3).ToArray(),
            Width = Size,
            Height = Size,
            Seed = 5,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase),
        };

    private static byte At(ImageResult image, int x, int y) => image.Rgb[(y * Size + x) * 3];

    /// <summary>SwarmUI's default: any mask value above ~0 takes the generated pixel in full, so a gray mask edge is a hard edge.</summary>
    [Fact]
    public void Apply_ByDefault_HardThresholdsTheMask()
    {
        ImageResult result = MaskRecomposite.Apply(Request(HalfMaskImage(HalfMask)), Result());

        Assert.Equal(Generated, At(result, 4, 10));
        Assert.Equal(Original, At(result, 28, 10));
        Assert.Equal(5, result.Seed);
    }

    [Fact]
    public void Apply_WhenUnthresholded_BlendsThroughTheSoftMask()
    {
        ImageResult result = MaskRecomposite.Apply(Request(HalfMaskImage(HalfMask), unthresholded: true), Result());

        byte blended = At(result, 4, 10);
        Assert.InRange(blended, Original + 40, Generated - 40);
        Assert.Equal(Original, At(result, 28, 10));
    }

    /// <summary>A mask paired with a reference-mode init is still pasted, as in SwarmUI; pinned so a family that starts consuming the mask itself is a deliberate change.</summary>
    [Fact]
    public void Apply_WithAReferenceModeInit_StillPastesThroughTheMask()
    {
        ImageRequest request = Request(HalfMaskImage(255)) with
        {
            Img2Img = new Img2Img { InitImage = Solid(Original), Mode = Img2ImgMode.Reference },
        };

        ImageResult result = MaskRecomposite.Apply(request, Result());

        Assert.Equal(Generated, At(result, 4, 10));
        Assert.Equal(Original, At(result, 28, 10));
    }

    [Fact]
    public void Apply_WhenRecompositeIsOff_ReturnsTheGeneratedImageAsIs()
    {
        ImageResult generated = Result();

        Assert.Same(generated, MaskRecomposite.Apply(Request(HalfMaskImage(255), recomposite: false), generated));
    }

    [Fact]
    public void Apply_WithoutAMask_ReturnsTheGeneratedImageAsIs()
    {
        ImageResult generated = Result();
        ImageRequest noMask = new ImageRequest { Prompt = "test", Img2Img = new Img2Img { InitImage = Solid(Original) } };

        Assert.Same(generated, MaskRecomposite.Apply(noMask, generated));
    }

    /// <summary>Grow/blur are applied before the threshold, so a blurred edge widens the hard-edged paste region.</summary>
    [Fact]
    public void Apply_UsesTheGrownMask()
    {
        ImageRequest request = Request(HalfMaskImage(255)) with
        {
            Inpaint = new Inpaint { Mask = HalfMaskImage(255), Grow = 8 },
        };

        ImageResult result = MaskRecomposite.Apply(request, Result());

        // Grow 8 expands 4px per side: column 16..19 now take the new pixel, column 20 does not.
        Assert.Equal(Generated, At(result, 19, 10));
        Assert.Equal(Original, At(result, 20, 10));
    }

    /// <summary>A family that snaps its dimensions returns a different size than the init image; the paste is made at the output size.</summary>
    [Fact]
    public void Apply_WhenTheOutputSizeDiffersFromTheInit_PastesAtTheOutputSize()
    {
        const int outSize = Size / 2;
        ImageResult smaller = new ImageResult
        {
            Rgb = Enumerable.Repeat(Generated, outSize * outSize * 3).ToArray(),
            Width = outSize,
            Height = outSize,
            Seed = 5,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase),
        };

        ImageResult result = MaskRecomposite.Apply(Request(HalfMaskImage(255)), smaller);

        Assert.Equal(outSize, result.Width);
        Assert.Equal(outSize * outSize * 3, result.Rgb.Length);
        Assert.Equal(Generated, result.Rgb[(4 * outSize + 2) * 3]);
        Assert.Equal(Original, result.Rgb[(4 * outSize + 13) * 3]);
    }

    /// <summary>An empty mask declines the crop; the fallback then resolves a mask instead of tripping the resolver's guard.</summary>
    [Fact]
    public void WithoutCrop_AfterADeclinedCropOnAnEmptyMask_ResolvesWithoutThrowing()
    {
        ImageRequest request = Request(Solid(0)) with
        {
            Inpaint = new Inpaint { Mask = Solid(0), ShrinkGrow = 8 },
        };

        Assert.Null(InpaintOnlyMasked.Prepare(request));
        ImageRequest fallback = InpaintOnlyMasked.WithoutCrop(request);

        Assert.NotNull(MaskResolver.ResolveBytes(fallback.Inpaint, Size, Size));
    }

    [Fact]
    public void BinarizeInPlace_MapsAtOrAboveTheThresholdToFullAndTheRestToZero()
    {
        byte[] mask = [0, 1, 2, 128, 255];

        FeatureImaging.BinarizeInPlace(mask, 1);

        Assert.Equal(new byte[] { 0, 255, 255, 255, 255 }, mask);
    }

    /// <summary>The full-canvas fallback after a declined crop must not hand the mask resolver a request that still asks to crop.</summary>
    [Theory]
    [InlineData(8, false)]
    [InlineData(0, true)]
    public void WithoutCrop_ClearsTheCropRequestAndKeepsTheMask(int shrinkGrow, bool cropToMask)
    {
        ImageRequest request = Request(HalfMaskImage(255)) with
        {
            Inpaint = new Inpaint { Mask = HalfMaskImage(255), ShrinkGrow = shrinkGrow, CropToMask = cropToMask, Grow = 3 },
        };

        ImageRequest result = InpaintOnlyMasked.WithoutCrop(request);

        Assert.False(result.Inpaint!.CropsToMask);
        Assert.Equal(3, result.Inpaint.Grow);
        Assert.Same(request.Inpaint!.Mask, result.Inpaint.Mask);
        Assert.NotNull(MaskResolver.ResolveBytes(result.Inpaint, Size, Size));
    }
}
