using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Recipes.Image;

/// <summary>Prepares the images of a Fibo 2.0 edit as the model was trained on them (ComfyUI <c>TextEncodeFibo2</c> and
/// <c>trained_size</c>, diffusers <c>BriaFibo2ImageProcessor</c>): a masked region greyed out first, a single image
/// cover-cropped to the result's size and several each to the size trained for their own aspect ratio, then two copies of each
/// crop: the VAE's, and the vision tower's at most 768 pixels a side on the tower's 32-pixel grid.</summary>
internal static class Fibo2EditImages
{
    /// <summary>Most images an edit takes (the card's <c>&lt;image_1&gt;</c>..<c>&lt;image_4&gt;</c>).</summary>
    public const int MaxImages = 4;

    private const int TrainedArea = 1024 * 1024;
    private const int Multiple = 16;
    private const int VisionMaxSide = 768;

    /// <summary>The size Fibo 2.0 was trained at for an aspect ratio: of the multiples of 16 within 10% of one megapixel, the
    /// one that crops least from a picture of that aspect, then the closest to one megapixel, then the wider.</summary>
    public static (int Width, int Height) TrainedSize(int imageWidth, int imageHeight)
    {
        if (imageWidth <= 0 || imageHeight <= 0)
            throw new ArgumentOutOfRangeException(nameof(imageWidth), $"Image size {imageWidth}x{imageHeight} must be positive.");
        double aspect = imageWidth / (double)imageHeight;
        (int Width, int Height) best = default;
        (double Crop, long AreaGap, int NegWidth) bestKey = (double.MaxValue, long.MaxValue, 0);
        int minSide = 4 * Multiple;
        for (int w = minSide; w < 2 * TrainedArea / minSide; w += Multiple)
        {
            int hLow = Math.Max(minSide, Multiple * (int)Math.Ceiling(0.9 * TrainedArea / w / Multiple));
            int hHigh = Multiple * (int)Math.Floor(1.1 * TrainedArea / w / Multiple);
            for (int h = hLow; h <= hHigh; h += Multiple)
            {
                double ratio = w / (double)h;
                double crop = Math.Round(1 - Math.Min(ratio / aspect, aspect / ratio), 9, MidpointRounding.ToEven);
                (double, long, int) key = (crop, Math.Abs((long)w * h - TrainedArea), -w);
                if (key.CompareTo(bestKey) < 0)
                {
                    bestKey = key;
                    best = (w, h);
                }
            }
        }
        return best;
    }

    /// <summary>The image with the region <paramref name="mask"/> marks (L8, 255 = regenerate, at the image's size) replaced by flat
    /// (128, 128, 128) grey through the mask made binary at 128, before any resize, as training did.</summary>
    public static ImageData GreyOut(ImageData image, byte[] mask)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(mask);
        long pixels = (long)image.Width * image.Height;
        if (mask.LongLength != pixels)
            throw new ArgumentException($"Mask has {mask.LongLength} samples for a {image.Width}x{image.Height} image.", nameof(mask));
        byte[] rgb = new byte[pixels * 3];
        Array.Copy(image.Rgb, rgb, rgb.Length);
        for (long i = 0; i < pixels; i++)
        {
            if (mask[i] < 128)
                continue;
            rgb[i * 3] = rgb[i * 3 + 1] = rgb[i * 3 + 2] = 128;
        }
        return new ImageData { Rgb = rgb, Width = image.Width, Height = image.Height };
    }

    /// <summary>The two copies of every image, sized for <paramref name="width"/> x <paramref name="height"/>, and the number of
    /// vision tokens each vision copy yields. Caller disposes.</summary>
    public static Prepared Prepare(IReadOnlyList<ImageData> images, int width, int height, Qwen3VlImageProcessor processor, int visionFactor)
    {
        ArgumentNullException.ThrowIfNull(images);
        ArgumentNullException.ThrowIfNull(processor);
        if (images.Count == 0)
            throw new ArgumentException("An edit needs at least one image.", nameof(images));
        List<Tensor> pixels = new(images.Count);
        List<Tensor> vision = new(images.Count);
        int[] padCounts = new int[images.Count];
        try
        {
            for (int i = 0; i < images.Count; i++)
            {
                (int cropW, int cropH) = images.Count == 1 ? (width, height) : TrainedSize(images[i].Width, images[i].Height);
                ImageData crop = FeatureImaging.CoverCrop(images[i], cropW, cropH);
                pixels.Add(FeatureImaging.RgbToTensorMinusOneOne(crop.Rgb, cropW, cropH));

                // Qwen3-VL sees each crop at most 768 pixels a side, then on its 32-pixel grid (ComfyUI thumbnail + to_qwen_grid).
                (int thumbW, int thumbH) = Thumbnail(cropW, cropH);
                ImageData thumb = thumbW == cropW && thumbH == cropH
                    ? crop : new ImageData { Rgb = FeatureImaging.ResizeRgb24(crop, thumbW, thumbH), Width = thumbW, Height = thumbH };
                (int gridH, int gridW) = processor.SmartResize(thumbH, thumbW);
                byte[] gridRgb = FeatureImaging.ResizeRgb24(thumb, gridW, gridH);
                vision.Add(ToChw(gridRgb, gridW, gridH));
                padCounts[i] = gridH / visionFactor * (gridW / visionFactor);
            }
        }
        catch
        {
            foreach (Tensor tensor in pixels) tensor.Dispose();
            foreach (Tensor tensor in vision) tensor.Dispose();
            throw;
        }
        return new Prepared(pixels, vision, padCounts);
    }

    private static (int Width, int Height) Thumbnail(int width, int height)
    {
        double scale = VisionMaxSide / (double)Math.Max(width, height);
        if (scale >= 1)
            return (width, height);
        return (Math.Max(1, (int)Math.Round(width * scale)), Math.Max(1, (int)Math.Round(height * scale)));
    }

    private static unsafe Tensor ToChw(byte[] rgb, int width, int height)
    {
        Tensor tensor = new(new TensorShape(3, height, width), DType.F32);
        float* p = (float*)tensor.DataPointer;
        int spatial = width * height;
        for (int c = 0; c < 3; c++)
        {
            for (int i = 0; i < spatial; i++)
                p[c * spatial + i] = rgb[i * 3 + c] / 255f;
        }
        return tensor;
    }

    /// <summary>The VAE pixel copies <c>[1, 3, H, W]</c> in [-1, 1], the vision copies <c>[3, H, W]</c> in [0, 1] and each vision
    /// copy's token count.</summary>
    internal sealed class Prepared(IReadOnlyList<Tensor> pixels, IReadOnlyList<Tensor> vision, int[] padCounts) : IDisposable
    {
        public IReadOnlyList<Tensor> Pixels { get; } = pixels;
        public IReadOnlyList<Tensor> Vision { get; } = vision;
        public int[] PadCounts { get; } = padCounts;

        public void Dispose()
        {
            foreach (Tensor tensor in Pixels) tensor.Dispose();
            foreach (Tensor tensor in Vision) tensor.Dispose();
        }
    }
}
