using HartsyInference.Core.Logging;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Features;

/// <summary>"Inpaint only masked" (<see cref="Inpaint.ShrinkGrow"/>): instead of denoising the whole canvas, crop the init image to the mask's bounding box grown by N pixels, generate at the model's native resolution over just that crop, then scale the result back and composite it into the original. The masked region therefore receives the model's full resolution budget, which is the entire point — refining a face on a 2K canvas at 1K native otherwise spends almost all of its pixels on the parts the user did not select. <para>This mirrors the node graph SwarmUI's own inpaint path builds (<c>SwarmMaskBounds</c> → <c>SwarmImageCrop</c> → <c>SwarmImageScaleForMP</c> → sample → <c>ImageScale</c> → <c>ImageCompositeMasked</c>), so an image generated here matches what the same settings produced before, rather than merely being plausible.</para> <para>It sits above the recipe pipelines because it changes the generated resolution and needs a post-generation composite — no pipeline can implement it alone.</para></summary>
public static class InpaintOnlyMasked
{
    /// <summary>Mask values below this are treated as unselected when finding the bounding box, matching the 0.01 threshold SwarmUI applies before <c>SwarmMaskBounds</c>.</summary>
    private const byte BoundsThreshold = 3;

    /// <summary>Applied to the cropped mask before compositing so a blurred edge cannot bleed the patch into pixels the user never selected (SwarmUI's <c>ThresholdMask</c> at 0.001).</summary>
    private const byte CompositeThreshold = 1;

    /// <summary>Generated dimensions are rounded to this so every family's own snapping is a no-op — the widest VAE downscale × patch factor in the engine is 16.</summary>
    private const int SizeAlignment = 16;

    /// <summary>The resolved crop: where it came from, what to generate, and what to composite back into.</summary>
    public sealed record Plan(int X, int Y, int CropWidth, int CropHeight, int GenerateWidth, int GenerateHeight,
        ImageData OriginalInit, ImageData CroppedInit, byte[] CroppedMask);

    /// <summary>Resolves the crop for <paramref name="request"/>, or null when this run is not an "inpaint only masked" one — no init image, no mask, <see cref="Inpaint.ShrinkGrow"/> at 0, or a mask that selects nothing. <paramref name="request"/> must already have its defaults applied, since the crop is scaled to the resolution the model will actually run at.</summary>
    public static Plan? Prepare(ImageRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Inpaint is null || request.Img2Img is null || request.Inpaint.ShrinkGrow == 0)
        {
            return null;
        }

        ImageData init = request.Img2Img.InitImage;
        // The mask is authored against the init image, so bounds are found at the init image's own resolution.
        byte[] mask = FeatureImaging.ResizeGrayscale(request.Inpaint.Mask, init.Width, init.Height);
        if (request.Inpaint.Grow > 0)
        {
            FeatureImaging.DilateInPlace(mask, init.Width, init.Height, request.Inpaint.Grow);
        }
        if (request.Inpaint.Blur > 0)
        {
            FeatureImaging.GaussianBlurInPlace(mask, init.Width, init.Height, request.Inpaint.Blur / 2.0f);
        }

        // ShrinkGrow < 0 is accepted as a shrink of the box; the bounds helper clamps either way.
        (int X, int Y, int Width, int Height)? bounds =
            FeatureImaging.MaskBounds(mask, init.Width, init.Height, request.Inpaint.ShrinkGrow, BoundsThreshold);
        if (bounds is null)
        {
            Logs.Warning("[Features][InpaintOnlyMasked] The mask selects no pixels; falling back to a full-canvas inpaint.");
            return null;
        }

        (int x, int y, int cropWidth, int cropHeight) = bounds.Value;
        if (cropWidth >= init.Width && cropHeight >= init.Height)
        {
            Logs.Info("[Features][InpaintOnlyMasked] The grown mask covers the whole image; running a full-canvas inpaint.");
            return null;
        }

        (int generateWidth, int generateHeight) = ScaleToBudget(cropWidth, cropHeight, request);
        ImageData croppedInit = FeatureImaging.CropRgb24(init, x, y, cropWidth, cropHeight);
        byte[] croppedMask = FeatureImaging.CropGray(mask, init.Width, init.Height, x, y, cropWidth, cropHeight);

        Logs.Info($"[Features][InpaintOnlyMasked] Crop {cropWidth}x{cropHeight} at ({x},{y}) of {init.Width}x{init.Height}, "
            + $"generating at {generateWidth}x{generateHeight} (grow={request.Inpaint.ShrinkGrow}px).");
        return new Plan(x, y, cropWidth, cropHeight, generateWidth, generateHeight, init, croppedInit, croppedMask);
    }

    /// <summary>Rewrites the request to generate the crop: the init image and mask become the cropped pair resized to the generation size, and <see cref="Inpaint.ShrinkGrow"/> is cleared so the downstream mask resolver treats this as an ordinary inpaint.</summary>
    public static ImageRequest Apply(ImageRequest request, Plan plan)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(plan);
        ImageData scaledInit = new ImageData
        {
            Rgb = FeatureImaging.ResizeRgb24(plan.CroppedInit, plan.GenerateWidth, plan.GenerateHeight),
            Width = plan.GenerateWidth,
            Height = plan.GenerateHeight,
        };
        ImageData maskImage = GrayToImage(plan.CroppedMask, plan.CropWidth, plan.CropHeight);
        return request with
        {
            Width = plan.GenerateWidth,
            Height = plan.GenerateHeight,
            Img2Img = request.Img2Img! with { InitImage = scaledInit },
            // Grow and Blur were already baked into the cropped mask; re-applying them here would double them.
            Inpaint = new Inpaint { Mask = maskImage },
        };
    }

    /// <summary>Scales the generated crop back to its original size and composites it into the untouched canvas, so every pixel outside the mask is bit-identical to the input.</summary>
    public static ImageResult Composite(ImageResult generated, Plan plan)
    {
        ArgumentNullException.ThrowIfNull(generated);
        ArgumentNullException.ThrowIfNull(plan);
        ImageData patch = new ImageData { Rgb = generated.Rgb, Width = generated.Width, Height = generated.Height };
        // Resized against the pipeline's *actual* output size, which a family that snaps its dimensions may have
        // changed from what Apply requested.
        ImageData scaledBack = new ImageData
        {
            Rgb = FeatureImaging.ResizeRgb24(patch, plan.CropWidth, plan.CropHeight),
            Width = plan.CropWidth,
            Height = plan.CropHeight,
        };
        byte[] compositeMask = (byte[])plan.CroppedMask.Clone();
        FeatureImaging.ThresholdInPlace(compositeMask, CompositeThreshold);
        ImageData composited = FeatureImaging.CompositeRgb24(plan.OriginalInit, scaledBack, compositeMask, plan.X, plan.Y);

        Dictionary<string, string> meta = new Dictionary<string, string>(generated.Meta, StringComparer.OrdinalIgnoreCase)
        {
            ["inpaint_only_masked"] = $"{plan.CropWidth}x{plan.CropHeight}+{plan.X}+{plan.Y}@{generated.Width}x{generated.Height}",
            ["size"] = $"{composited.Width}x{composited.Height}",
        };
        return new ImageResult
        {
            Rgb = composited.Rgb,
            Width = composited.Width,
            Height = composited.Height,
            Seed = generated.Seed,
            Meta = meta,
        };
    }

    /// <summary>Scales the crop to the request's pixel budget while preserving its aspect ratio — the engine's equivalent of <c>SwarmImageScaleForMP</c> with <c>can_shrink</c>, so a small crop is upscaled into the model's native resolution and an oversized one is brought back down.</summary>
    private static (int Width, int Height) ScaleToBudget(int cropWidth, int cropHeight, ImageRequest request)
    {
        (int targetWidth, int targetHeight) = RecipeRequestMapper.Size(request);
        double budget = (double)targetWidth * targetHeight;
        double scale = Math.Sqrt(budget / ((double)cropWidth * cropHeight));
        int width = Align((int)Math.Round(cropWidth * scale));
        int height = Align((int)Math.Round(cropHeight * scale));
        return (width, height);
    }

    private static int Align(int value) =>
        Math.Max(SizeAlignment, (value + SizeAlignment / 2) / SizeAlignment * SizeAlignment);

    /// <summary>Wraps an L8 buffer as an <see cref="ImageData"/>, since the composition contract carries masks as images and the resolver collapses them back to luminance.</summary>
    private static ImageData GrayToImage(byte[] gray, int width, int height)
    {
        byte[] rgb = new byte[(long)width * height * 3];
        for (long i = 0; i < gray.LongLength; i++)
        {
            rgb[i * 3] = gray[i];
            rgb[i * 3 + 1] = gray[i];
            rgb[i * 3 + 2] = gray[i];
        }
        return new ImageData { Rgb = rgb, Width = width, Height = height };
    }
}
