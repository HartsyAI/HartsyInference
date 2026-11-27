using HartsyInference.Core.Imaging;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>One image run through the Hy4 vision preprocessing: its patch rows, the grid they form, and the size the image was resized to.</summary>
/// <param name="PixelValues">Patch rows, <c>[GridH * GridW, 768]</c> row-major, each row channel-major (C, patch rows, patch columns).</param>
/// <param name="GridH">Patch rows of the resized image (before the 2x2 merge).</param>
/// <param name="GridW">Patch columns of the resized image.</param>
/// <param name="ResizedHeight">Height the image was resized to, a multiple of the patch times the merge.</param>
/// <param name="ResizedWidth">Width the image was resized to.</param>
public sealed record Hy4FlashPatchedImage(float[] PixelValues, int GridH, int GridW, int ResizedHeight, int ResizedWidth);

/// <summary>The Hy4 vision image preprocessing: the released <c>HYV4VLImageProcessor</c> (transformers eb1d5db), for one image.</summary>
/// <remarks>Smart resize to a multiple of patch times merge (32) inside the released pixel bounds, rescale by 1/255, normalise with mean and std 0.5,
/// and flatten into patch rows in row-major grid order. The resize is the Core bicubic, not the processor's, so only an image already on the grid (no resize)
/// matches the processor's values exactly; an image that needs a resize gets the grid and the size exactly and values within the resampler's difference.</remarks>
public static class Hy4FlashImagePreprocessor
{
    /// <summary>Patch edge in pixels, from the released <c>preprocessor_config.json</c> (<c>patch_size</c>).</summary>
    public const int PatchSize = 16;

    /// <summary>Patches per merged edge (<c>merge_size</c>).</summary>
    public const int MergeSize = 2;

    /// <summary>The lower pixel bound (<c>min_pixels</c>).</summary>
    public const long MinPixels = 4096;

    /// <summary>The upper pixel bound (<c>max_pixels</c>).</summary>
    public const long MaxPixels = 4194304;

    /// <summary>Feature width of one patch: 3 channels times the patch area (one frame).</summary>
    public const int FeatureDim = 3 * PatchSize * PatchSize;

    private const float RescaleFactor = 0.00392156862745098f;
    private const float Mean = 0.5f;
    private const float Std = 0.5f;

    /// <summary>The processor's <c>smart_resize</c>: the size inside the pixel bounds, both edges divisible by <paramref name="factor"/>, closest to the
    /// input's aspect ratio. Integer rounding follows Python's <c>round</c> (ties to even).</summary>
    /// <exception cref="ArgumentOutOfRangeException">A side is not positive.</exception>
    /// <exception cref="ArgumentException">The aspect ratio is above 200.</exception>
    public static (int Height, int Width) SmartResize(int height, int width, int factor, long minPixels, long maxPixels)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        if ((double)Math.Max(height, width) / Math.Min(height, width) > 200)
            throw new ArgumentException($"The absolute aspect ratio must be at most 200, got {(double)Math.Max(height, width) / Math.Min(height, width)}.");
        int hBar = (int)(Math.Round(height / (double)factor, MidpointRounding.ToEven) * factor);
        int wBar = (int)(Math.Round(width / (double)factor, MidpointRounding.ToEven) * factor);
        if ((long)hBar * wBar > maxPixels)
        {
            double beta = Math.Sqrt((double)height * width / maxPixels);
            hBar = Math.Max(factor, (int)Math.Floor(height / beta / factor) * factor);
            wBar = Math.Max(factor, (int)Math.Floor(width / beta / factor) * factor);
        }
        else if ((long)hBar * wBar < minPixels)
        {
            double beta = Math.Sqrt(minPixels / ((double)height * width));
            hBar = (int)Math.Ceiling(height * beta / factor) * factor;
            wBar = (int)Math.Ceiling(width * beta / factor) * factor;
        }
        return (hBar, wBar);
    }

    /// <summary>Resizes one interleaved 8-bit RGB image (row-major HWC) with the processor's bounds, normalises it and flattens it into patch rows.</summary>
    /// <param name="rgb">The pixels, <c>height * width * 3</c> bytes.</param>
    /// <param name="width">Image width.</param>
    /// <param name="height">Image height.</param>
    public static Hy4FlashPatchedImage Preprocess(ReadOnlySpan<byte> rgb, int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        if (rgb.Length != checked(width * height * 3)) throw new ArgumentException($"Expected {width * height * 3} bytes of RGB, got {rgb.Length}.", nameof(rgb));
        (int rh, int rw) = SmartResize(height, width, PatchSize * MergeSize, MinPixels, MaxPixels);
        int gridH = rh / PatchSize, gridW = rw / PatchSize;

        // Resized pixels as floats in 0..255: the input itself when no resize is needed, else the bicubic resample quantised like an 8-bit image.
        float[] pixels = new float[rh * rw * 3];
        if (rh == height && rw == width)
        {
            for (int i = 0; i < pixels.Length; i++) pixels[i] = rgb[i];
        }
        else
        {
            Resample.BicubicHwc8(rgb, width, height, 3, pixels, rw, rh, a: -0.5f, antialias: true, quantizeU8: true);
        }

        float[] values = new float[gridH * gridW * FeatureDim];
        for (int gy = 0; gy < gridH; gy++)
        for (int gx = 0; gx < gridW; gx++)
        {
            int rowBase = (gy * gridW + gx) * FeatureDim;
            for (int c = 0; c < 3; c++)
            for (int py = 0; py < PatchSize; py++)
            for (int px = 0; px < PatchSize; px++)
            {
                int srcY = gy * PatchSize + py, srcX = gx * PatchSize + px;
                float scaled = pixels[(srcY * rw + srcX) * 3 + c] * RescaleFactor;
                values[rowBase + c * PatchSize * PatchSize + py * PatchSize + px] = (scaled - Mean) / Std;
            }
        }
        return new Hy4FlashPatchedImage(values, gridH, gridW, rh, rw);
    }
}
