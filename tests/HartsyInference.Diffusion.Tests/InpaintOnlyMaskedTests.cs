using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Behaviour of <see cref="InpaintOnlyMasked"/> — the crop/generate/composite wrapper that implements
/// <see cref="Inpaint.ShrinkGrow"/>. Weight-free: every assertion is about raster bookkeeping, which is where this
/// feature can silently go wrong (an off-by-one crop or a missed threshold shifts or bleeds the patch).</summary>
public sealed class InpaintOnlyMaskedTests
{
    private const int CanvasWidth = 256;
    private const int CanvasHeight = 192;

    private static ImageData SolidImage(int width, int height, byte value) =>
        new ImageData { Rgb = Enumerable.Repeat(value, width * height * 3).ToArray(), Width = width, Height = height };

    /// <summary>A black mask with one white rectangle — the region the user "painted".</summary>
    private static ImageData MaskWithRect(int x, int y, int width, int height)
    {
        byte[] rgb = new byte[CanvasWidth * CanvasHeight * 3];
        for (int row = y; row < y + height; row++)
        {
            for (int col = x; col < x + width; col++)
            {
                int idx = (row * CanvasWidth + col) * 3;
                rgb[idx] = rgb[idx + 1] = rgb[idx + 2] = 255;
            }
        }
        return new ImageData { Rgb = rgb, Width = CanvasWidth, Height = CanvasHeight };
    }

    private static ImageRequest RequestWith(ImageData mask, int shrinkGrow, byte initValue = 40, int grow = 0, bool crop = false) =>
        new ImageRequest
        {
            Prompt = "test",
            Width = 512,
            Height = 512,
            Img2Img = new Img2Img { InitImage = SolidImage(CanvasWidth, CanvasHeight, initValue) },
            Inpaint = new Inpaint { Mask = mask, ShrinkGrow = shrinkGrow, Grow = grow, CropToMask = crop },
        };

    [Fact]
    public void Prepare_WithoutShrinkGrow_ReturnsNull()
    {
        Assert.Null(InpaintOnlyMasked.Prepare(RequestWith(MaskWithRect(64, 48, 32, 24), shrinkGrow: 0)));
    }

    /// <summary>SwarmUI's Mask Shrink Grow is a toggle: on at 0 it crops exactly to the mask, it does not mean off.</summary>
    [Fact]
    public void Prepare_WithCropToMaskAndZeroShrinkGrow_CropsTightToTheMask()
    {
        InpaintOnlyMasked.Plan? plan = InpaintOnlyMasked.Prepare(RequestWith(MaskWithRect(64, 48, 32, 24), shrinkGrow: 0, crop: true));

        Assert.NotNull(plan);
        Assert.Equal(64, plan!.X);
        Assert.Equal(48, plan.Y);
        Assert.Equal(32, plan.CropWidth);
        Assert.Equal(24, plan.CropHeight);
    }

    /// <summary>Mask Grow matches SwarmMaskGrow: about half the stated pixels per side, not the full amount.</summary>
    [Theory]
    [InlineData(2, 1)]
    [InlineData(9, 5)]
    public void Prepare_MaskGrowExpandsTheMaskByHalfTheValuePerSide(int grow, int expectedPerSide)
    {
        InpaintOnlyMasked.Plan? plan = InpaintOnlyMasked.Prepare(
            RequestWith(MaskWithRect(64, 48, 32, 24), shrinkGrow: 0, grow: grow, crop: true));

        Assert.NotNull(plan);
        Assert.Equal(64 - expectedPerSide, plan!.X);
        Assert.Equal(48 - expectedPerSide, plan.Y);
        Assert.Equal(32 + 2 * expectedPerSide, plan.CropWidth);
        Assert.Equal(24 + 2 * expectedPerSide, plan.CropHeight);
    }

    /// <summary>An all-black mask selects nothing; falling back to the full canvas beats cropping to an empty box.</summary>
    [Fact]
    public void Prepare_WithEmptyMask_FallsBackToFullCanvas()
    {
        Assert.Null(InpaintOnlyMasked.Prepare(RequestWith(SolidImage(CanvasWidth, CanvasHeight, 0), shrinkGrow: 8)));
    }

    [Fact]
    public void Prepare_CropsToTheMaskBoundsGrownByShrinkGrow()
    {
        InpaintOnlyMasked.Plan? plan = InpaintOnlyMasked.Prepare(RequestWith(MaskWithRect(64, 48, 32, 24), shrinkGrow: 8));

        Assert.NotNull(plan);
        Assert.Equal(56, plan!.X);
        Assert.Equal(40, plan.Y);
        Assert.Equal(48, plan.CropWidth);
        Assert.Equal(40, plan.CropHeight);
        Assert.Equal(plan.CropWidth * plan.CropHeight, plan.CroppedMask.Length);
    }

    /// <summary>The grow is clamped to the canvas rather than producing a crop that starts off-image.</summary>
    [Fact]
    public void Prepare_ClampsTheGrownBoxToTheCanvas()
    {
        InpaintOnlyMasked.Plan? plan = InpaintOnlyMasked.Prepare(RequestWith(MaskWithRect(0, 0, 16, 16), shrinkGrow: 64));

        Assert.NotNull(plan);
        Assert.Equal(0, plan!.X);
        Assert.Equal(0, plan.Y);
        Assert.True(plan.X + plan.CropWidth <= CanvasWidth);
        Assert.True(plan.Y + plan.CropHeight <= CanvasHeight);
    }

    /// <summary>The whole point of the feature: a small crop is scaled up to the model's pixel budget so the masked
    /// region gets the full resolution, rather than being generated at its tiny native size.</summary>
    [Fact]
    public void Prepare_ScalesTheCropUpToTheRequestedPixelBudget()
    {
        InpaintOnlyMasked.Plan? plan = InpaintOnlyMasked.Prepare(RequestWith(MaskWithRect(64, 48, 32, 24), shrinkGrow: 8));

        Assert.NotNull(plan);
        long budget = 512L * 512;
        long generated = (long)plan!.GenerateWidth * plan.GenerateHeight;
        Assert.InRange(generated, (long)(budget * 0.85), (long)(budget * 1.15));
        Assert.True(plan.GenerateWidth > plan.CropWidth, "a 48px crop should be upscaled toward the model's native size");
        Assert.Equal(0, plan.GenerateWidth % 16);
        Assert.Equal(0, plan.GenerateHeight % 16);
        // Aspect ratio is preserved so the crop is not distorted before generation.
        double cropAspect = (double)plan.CropWidth / plan.CropHeight;
        double genAspect = (double)plan.GenerateWidth / plan.GenerateHeight;
        Assert.InRange(genAspect, cropAspect * 0.9, cropAspect * 1.1);
    }

    [Fact]
    public void Apply_RewritesTheRequestToGenerateTheCropAndClearsShrinkGrow()
    {
        ImageRequest request = RequestWith(MaskWithRect(64, 48, 32, 24), shrinkGrow: 8);
        InpaintOnlyMasked.Plan plan = InpaintOnlyMasked.Prepare(request)!;

        ImageRequest applied = InpaintOnlyMasked.Apply(request, plan);

        Assert.Equal(plan.GenerateWidth, applied.Width);
        Assert.Equal(plan.GenerateHeight, applied.Height);
        Assert.Equal(plan.GenerateWidth, applied.Img2Img!.InitImage.Width);
        Assert.Equal(plan.GenerateHeight, applied.Img2Img.InitImage.Height);
        Assert.Equal(plan.CropWidth, applied.Inpaint!.Mask.Width);
        // Cleared so the downstream resolver treats it as an ordinary inpaint, and re-running grow/blur cannot double it.
        Assert.Equal(0, applied.Inpaint.ShrinkGrow);
        Assert.False(applied.Inpaint.CropToMask);
        Assert.Equal(0, applied.Inpaint.Grow);
        Assert.Equal(0, applied.Inpaint.Blur);
        Assert.Equal("test", applied.Prompt);
    }

    /// <summary>The guarantee that makes this safe to enable by default: everything outside the mask survives
    /// bit-identical, so repeated inpaints cannot accumulate VAE round-trip drift across the whole canvas.</summary>
    [Fact]
    public void Composite_LeavesEveryPixelOutsideTheMaskUntouched()
    {
        ImageRequest request = RequestWith(MaskWithRect(64, 48, 32, 24), shrinkGrow: 8, initValue: 40);
        InpaintOnlyMasked.Plan plan = InpaintOnlyMasked.Prepare(request)!;
        ImageResult generated = new ImageResult
        {
            Rgb = Enumerable.Repeat((byte)200, plan.GenerateWidth * plan.GenerateHeight * 3).ToArray(),
            Width = plan.GenerateWidth,
            Height = plan.GenerateHeight,
            Seed = 7,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase),
        };

        ImageResult composited = InpaintOnlyMasked.Composite(generated, plan);

        Assert.Equal(CanvasWidth, composited.Width);
        Assert.Equal(CanvasHeight, composited.Height);
        Assert.Equal(7, composited.Seed);
        for (int y = 0; y < CanvasHeight; y++)
        {
            for (int x = 0; x < CanvasWidth; x++)
            {
                bool insideMask = x >= 64 && x < 96 && y >= 48 && y < 72;
                byte actual = composited.Rgb[(y * CanvasWidth + x) * 3];
                if (insideMask)
                {
                    Assert.True(actual > 150, $"masked pixel ({x},{y}) should carry the generated patch; got {actual}");
                }
                else
                {
                    Assert.True(actual == 40, $"unmasked pixel ({x},{y}) must be untouched; got {actual}");
                }
            }
        }
    }

    /// <summary>A family that snaps its dimensions returns a different size than requested; the composite must scale
    /// from what the pipeline actually produced, not from what was asked for.</summary>
    [Fact]
    public void Composite_HandlesAPipelineThatReturnedADifferentSize()
    {
        ImageRequest request = RequestWith(MaskWithRect(64, 48, 32, 24), shrinkGrow: 8);
        InpaintOnlyMasked.Plan plan = InpaintOnlyMasked.Prepare(request)!;
        int snappedWidth = plan.GenerateWidth - 16;
        int snappedHeight = plan.GenerateHeight - 16;
        ImageResult generated = new ImageResult
        {
            Rgb = Enumerable.Repeat((byte)200, snappedWidth * snappedHeight * 3).ToArray(),
            Width = snappedWidth,
            Height = snappedHeight,
            Seed = 3,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase),
        };

        ImageResult composited = InpaintOnlyMasked.Composite(generated, plan);

        Assert.Equal(CanvasWidth, composited.Width);
        Assert.Equal(CanvasHeight, composited.Height);
        Assert.Contains("inpaint_only_masked", composited.Meta.Keys);
    }

    /// <summary>Default composite is a hard edge (SwarmUI's ThresholdMask); a blurred mask only softens the denoise, not the final paste.</summary>
    [Fact]
    public void Composite_ByDefault_PastesTheBlurredEdgeAtFullStrength()
    {
        ImageRequest hardRequest = RequestWith(MaskWithRect(64, 48, 32, 24), shrinkGrow: 8, grow: 0) with
        {
            Inpaint = new Inpaint { Mask = MaskWithRect(64, 48, 32, 24), ShrinkGrow = 8, Blur = 8 },
        };
        ImageRequest softRequest = hardRequest with { MaskCompositeUnthresholded = true };

        byte hardEdge = EdgePixelAfterComposite(hardRequest);
        byte softEdge = EdgePixelAfterComposite(softRequest);

        Assert.Equal(200, hardEdge);
        Assert.InRange(softEdge, 41, 199);
    }

    /// <summary>The pixel one step outside the painted rect, where a blur leaves a weak but nonzero mask value.</summary>
    private static byte EdgePixelAfterComposite(ImageRequest request)
    {
        InpaintOnlyMasked.Plan plan = InpaintOnlyMasked.Prepare(request)!;
        ImageResult generated = new ImageResult
        {
            Rgb = Enumerable.Repeat((byte)200, plan.GenerateWidth * plan.GenerateHeight * 3).ToArray(),
            Width = plan.GenerateWidth,
            Height = plan.GenerateHeight,
            Seed = 1,
            Meta = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase),
        };
        ImageResult composited = InpaintOnlyMasked.Composite(generated, plan);
        return composited.Rgb[(60 * CanvasWidth + 62) * 3];
    }
}
