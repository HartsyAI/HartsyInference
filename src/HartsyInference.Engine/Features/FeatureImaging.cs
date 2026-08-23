using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Requests;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Engine.Features;

/// <summary>Host-image-library-free raster helpers the feature resolvers share: bicubic resize of an <see cref="ImageData"/>, grayscale extraction for masks, separable dilation/blur, and the RGB→tensor packings the diffusion pipelines expect. Replaces the SwarmUI extension's ImageSharp calls with the engine's own <see cref="Resample"/> kernel (Keys bicubic, <c>a = -0.5</c>, antialiased — the PIL/torch convention).</summary>
public static class FeatureImaging
{
    /// <summary>Bicubic-resizes an image to interleaved HWC RGB24 at the target size; a same-size image is copied verbatim.</summary>
    public static byte[] ResizeRgb24(ImageData image, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (width <= 0 || height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width), $"Target size must be positive; got {width}x{height}.");
        }
        long expected = (long)image.Width * image.Height * 3;
        if (image.Rgb.LongLength < expected)
        {
            throw new InvalidOperationException($"Image has {image.Rgb.LongLength} bytes, expected {expected} for {image.Width}x{image.Height} RGB24.");
        }
        if (image.Width == width && image.Height == height)
        {
            byte[] copy = new byte[expected];
            Array.Copy(image.Rgb, copy, expected);
            return copy;
        }
        float[] resized = new float[(long)width * height * 3];
        Resample.BicubicHwc8(image.Rgb, image.Width, image.Height, 3, resized, width, height, a: -0.5f, antialias: true);
        byte[] dst = new byte[resized.Length];
        for (int i = 0; i < resized.Length; i++)
        {
            dst[i] = (byte)Math.Clamp(MathF.Round(resized[i]), 0f, 255f);
        }
        return dst;
    }

    /// <summary>Bicubic-resizes an image and collapses it to a single-channel L8 buffer via ITU-R BT.709 luminance — the right semantic when a mask arrives as a colored paint layer.</summary>
    public static byte[] ResizeGrayscale(ImageData image, int width, int height)
    {
        byte[] rgb = ResizeRgb24(image, width, height);
        byte[] gray = new byte[(long)width * height];
        for (int i = 0; i < gray.Length; i++)
        {
            float lum = 0.2126f * rgb[i * 3] + 0.7152f * rgb[i * 3 + 1] + 0.0722f * rgb[i * 3 + 2];
            gray[i] = (byte)Math.Clamp(MathF.Round(lum), 0f, 255f);
        }
        return gray;
    }

    /// <summary>Tightest box containing every mask pixel at or above <paramref name="threshold"/>, expanded by <paramref name="grow"/> pixels and clamped to the image. Returns null when the mask is entirely below the threshold — there is nothing to inpaint, and the caller must fall back to the full canvas.</summary>
    public static (int X, int Y, int Width, int Height)? MaskBounds(byte[] mask, int width, int height, int grow, byte threshold)
    {
        ArgumentNullException.ThrowIfNull(mask);
        int minX = width, minY = height, maxX = -1, maxY = -1;
        for (int y = 0; y < height; y++)
        {
            int row = y * width;
            for (int x = 0; x < width; x++)
            {
                if (mask[row + x] < threshold)
                {
                    continue;
                }
                if (x < minX) { minX = x; }
                if (x > maxX) { maxX = x; }
                if (y < minY) { minY = y; }
                if (y > maxY) { maxY = y; }
            }
        }
        if (maxX < 0)
        {
            return null;
        }
        minX = Math.Max(0, minX - grow);
        minY = Math.Max(0, minY - grow);
        maxX = Math.Min(width - 1, maxX + grow);
        maxY = Math.Min(height - 1, maxY + grow);
        return (minX, minY, maxX - minX + 1, maxY - minY + 1);
    }

    /// <summary>Copies an axis-aligned RGB24 sub-rectangle out of <paramref name="image"/>.</summary>
    public static ImageData CropRgb24(ImageData image, int x, int y, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(image);
        ValidateCrop(image.Width, image.Height, x, y, width, height);
        byte[] dst = new byte[(long)width * height * 3];
        for (int row = 0; row < height; row++)
        {
            Array.Copy(image.Rgb, ((long)(y + row) * image.Width + x) * 3, dst, (long)row * width * 3, (long)width * 3);
        }
        return new ImageData { Rgb = dst, Width = width, Height = height };
    }

    /// <summary>Copies an axis-aligned sub-rectangle out of a single-channel L8 buffer.</summary>
    public static byte[] CropGray(byte[] gray, int sourceWidth, int sourceHeight, int x, int y, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(gray);
        ValidateCrop(sourceWidth, sourceHeight, x, y, width, height);
        byte[] dst = new byte[(long)width * height];
        for (int row = 0; row < height; row++)
        {
            Array.Copy(gray, (long)(y + row) * sourceWidth + x, dst, (long)row * width, width);
        }
        return dst;
    }

    /// <summary>Alpha-composites <paramref name="patch"/> over <paramref name="canvas"/> at <paramref name="x"/>, <paramref name="y"/>, weighted per pixel by <paramref name="mask"/> (255 = take the patch, 0 = keep the canvas). Returns a new image; neither input is mutated.</summary>
    public static ImageData CompositeRgb24(ImageData canvas, ImageData patch, byte[] mask, int x, int y)
    {
        ArgumentNullException.ThrowIfNull(canvas);
        ArgumentNullException.ThrowIfNull(patch);
        ArgumentNullException.ThrowIfNull(mask);
        ValidateCrop(canvas.Width, canvas.Height, x, y, patch.Width, patch.Height);
        if (mask.LongLength < (long)patch.Width * patch.Height)
        {
            throw new ArgumentException($"Mask has {mask.LongLength} entries, expected {(long)patch.Width * patch.Height} for {patch.Width}x{patch.Height}.", nameof(mask));
        }
        byte[] dst = new byte[canvas.Rgb.LongLength];
        Array.Copy(canvas.Rgb, dst, canvas.Rgb.LongLength);
        for (int row = 0; row < patch.Height; row++)
        {
            long canvasBase = ((long)(y + row) * canvas.Width + x) * 3;
            long patchBase = (long)row * patch.Width * 3;
            long maskBase = (long)row * patch.Width;
            for (int col = 0; col < patch.Width; col++)
            {
                float weight = mask[maskBase + col] / 255f;
                if (weight <= 0f)
                {
                    continue;
                }
                for (int channel = 0; channel < 3; channel++)
                {
                    long dstIdx = canvasBase + col * 3 + channel;
                    float blended = patch.Rgb[patchBase + col * 3 + channel] * weight + dst[dstIdx] * (1f - weight);
                    dst[dstIdx] = (byte)Math.Clamp(MathF.Round(blended), 0f, 255f);
                }
            }
        }
        return new ImageData { Rgb = dst, Width = canvas.Width, Height = canvas.Height };
    }

    /// <summary>Zeroes every mask pixel below <paramref name="threshold"/>, so a blurred edge does not bleed the patch into pixels the user never selected.</summary>
    public static void ThresholdInPlace(byte[] mask, byte threshold)
    {
        ArgumentNullException.ThrowIfNull(mask);
        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i] < threshold)
            {
                mask[i] = 0;
            }
        }
    }

    private static void ValidateCrop(int sourceWidth, int sourceHeight, int x, int y, int width, int height)
    {
        if (width <= 0 || height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width), $"Crop size must be positive; got {width}x{height}.");
        }
        if (x < 0 || y < 0 || x + width > sourceWidth || y + height > sourceHeight)
        {
            throw new ArgumentOutOfRangeException(nameof(x), $"Crop {width}x{height} at ({x},{y}) falls outside {sourceWidth}x{sourceHeight}.");
        }
    }

    /// <summary>Separable max-filter dilation over a Chebyshev window of <paramref name="radius"/>: two 1-D passes, O(W·H·R) instead of O(W·H·R²). Approximates a square morphological dilation — close enough for mask edges.</summary>
    public static void DilateInPlace(byte[] mask, int width, int height, int radius)
    {
        ArgumentNullException.ThrowIfNull(mask);
        if (radius <= 0)
        {
            return;
        }
        byte[] tmp = new byte[mask.Length];
        for (int y = 0; y < height; y++)
        {
            int rowOff = y * width;
            for (int x = 0; x < width; x++)
            {
                int x0 = Math.Max(0, x - radius);
                int x1 = Math.Min(width - 1, x + radius);
                byte mx = 0;
                for (int xi = x0; xi <= x1; xi++)
                {
                    byte v = mask[rowOff + xi];
                    if (v > mx)
                    {
                        mx = v;
                    }
                }
                tmp[rowOff + x] = mx;
            }
        }
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                int y0 = Math.Max(0, y - radius);
                int y1 = Math.Min(height - 1, y + radius);
                byte mx = 0;
                for (int yi = y0; yi <= y1; yi++)
                {
                    byte v = tmp[yi * width + x];
                    if (v > mx)
                    {
                        mx = v;
                    }
                }
                mask[y * width + x] = mx;
            }
        }
    }

    /// <summary>Separable Gaussian blur of a single-channel buffer with edge-clamped taps, kernel radius <c>ceil(3σ)</c>.</summary>
    public static void GaussianBlurInPlace(byte[] mask, int width, int height, float sigma)
    {
        ArgumentNullException.ThrowIfNull(mask);
        if (sigma < 1e-3f)
        {
            return;
        }
        int radius = Math.Max(1, (int)MathF.Ceiling(sigma * 3f));
        float[] kernel = new float[radius * 2 + 1];
        float twoSigmaSq = 2f * sigma * sigma;
        float sum = 0f;
        for (int i = -radius; i <= radius; i++)
        {
            float w = MathF.Exp(-(i * i) / twoSigmaSq);
            kernel[i + radius] = w;
            sum += w;
        }
        for (int i = 0; i < kernel.Length; i++)
        {
            kernel[i] /= sum;
        }
        float[] tmp = new float[mask.Length];
        for (int y = 0; y < height; y++)
        {
            int rowOff = y * width;
            for (int x = 0; x < width; x++)
            {
                float acc = 0f;
                for (int k = -radius; k <= radius; k++)
                {
                    acc += kernel[k + radius] * mask[rowOff + Math.Clamp(x + k, 0, width - 1)];
                }
                tmp[rowOff + x] = acc;
            }
        }
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                float acc = 0f;
                for (int k = -radius; k <= radius; k++)
                {
                    acc += kernel[k + radius] * tmp[Math.Clamp(y + k, 0, height - 1) * width + x];
                }
                mask[y * width + x] = (byte)Math.Clamp(MathF.Round(acc), 0f, 255f);
            }
        }
    }

    /// <summary>Packs an interleaved RGB24 buffer into a <c>[1, 3, H, W]</c> F32 tensor in <c>[0, 1]</c>.</summary>
    public static unsafe Tensor RgbToTensorZeroOne(ReadOnlySpan<byte> rgb, int width, int height)
    {
        Tensor output = new Tensor(new TensorShape(1, 3, height, width), DType.F32);
        float* dp = (float*)output.DataPointer;
        int spatial = width * height;
        for (int c = 0; c < 3; c++)
        {
            long chOff = (long)c * spatial;
            for (int i = 0; i < spatial; i++)
            {
                dp[chOff + i] = rgb[i * 3 + c] / 255f;
            }
        }
        return output;
    }

    /// <summary>Packs an interleaved RGB24 buffer into a <c>[1, 3, H, W]</c> F32 tensor in <c>[-1, 1]</c>.</summary>
    public static unsafe Tensor RgbToTensorMinusOneOne(ReadOnlySpan<byte> rgb, int width, int height)
    {
        Tensor output = new Tensor(new TensorShape(1, 3, height, width), DType.F32);
        float* dp = (float*)output.DataPointer;
        int spatial = width * height;
        const float Inv127_5 = 1f / 127.5f;
        for (int c = 0; c < 3; c++)
        {
            long chOff = (long)c * spatial;
            for (int i = 0; i < spatial; i++)
            {
                dp[chOff + i] = rgb[i * 3 + c] * Inv127_5 - 1f;
            }
        }
        return output;
    }

    /// <summary>Packs a single-channel L8 buffer into a <c>[1, 1, H, W]</c> F32 tensor normalized to <c>[0, 1]</c>.</summary>
    public static unsafe Tensor GrayToMaskTensor(byte[] gray, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(gray);
        Tensor tensor = new Tensor(new TensorShape(1, 1, height, width), DType.F32);
        float* dp = (float*)tensor.DataPointer;
        const float Inv255 = 1f / 255f;
        for (int i = 0; i < gray.Length; i++)
        {
            dp[i] = gray[i] * Inv255;
        }
        return tensor;
    }
}
