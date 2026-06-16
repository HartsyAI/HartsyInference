using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Prompting;

/// <summary>A region's spatial extent for regional conditioning: either an axis-aligned <see cref="RectMask"/> (bbox case) or a soft per-pixel <see cref="Bitmap"/> (segmentation case). Both rasterize to a latent-resolution weight mask via <see cref="ToLatentMask"/>, so rectangle and arbitrary-mask use share one downstream path. <see cref="Width"/>/<see cref="Height"/> are the reference resolution the rect or bitmap is defined against.</summary>
public readonly record struct RegionMask
{
    /// <summary>The rectangle, in pixel space relative to <see cref="Width"/>/<see cref="Height"/>. Null when this is a bitmap mask.</summary>
    public RectMask? Rect { get; init; }

    /// <summary>Row-major <c>[Height*Width]</c> values in <c>[0, 1]</c>. Null when this is a rectangle mask.</summary>
    public float[]? Bitmap { get; init; }

    /// <summary>Reference width of the rect/bitmap coordinate space.</summary>
    public int Width { get; init; }

    /// <summary>Reference height of the rect/bitmap coordinate space.</summary>
    public int Height { get; init; }

    /// <summary>Creates a rectangle region at the given reference resolution.</summary>
    public static RegionMask FromRect(RectMask rect, int width, int height)
    {
        if (width <= 0 || height <= 0)
        {
            throw new HartsyInferenceException($"RegionMask resolution must be positive; got {width}x{height}.");
        }
        return new RegionMask { Rect = rect, Width = width, Height = height };
    }

    /// <summary>Creates a soft bitmap region; <paramref name="bitmap"/> length must equal <paramref name="width"/>*<paramref name="height"/>.</summary>
    public static RegionMask FromBitmap(float[] bitmap, int width, int height)
    {
        if (bitmap is null)
        {
            throw new ArgumentNullException(nameof(bitmap));
        }
        if (width <= 0 || height <= 0)
        {
            throw new HartsyInferenceException($"RegionMask resolution must be positive; got {width}x{height}.");
        }
        if (bitmap.Length != (long)width * height)
        {
            throw new HartsyInferenceException($"Bitmap length {bitmap.Length} must equal {width}*{height}={(long)width * height}.");
        }
        return new RegionMask { Bitmap = bitmap, Width = width, Height = height };
    }

    /// <summary>Rasterizes the region to its reference resolution then resamples (area-average) to a <c>[1, 1, latentH, latentW]</c> F32 mask in <c>[0, 1]</c>. Built once at plan time, not on the hot path.</summary>
    public unsafe Tensor ToLatentMask(int latentH, int latentW)
    {
        if (latentH <= 0 || latentW <= 0)
        {
            throw new HartsyInferenceException($"Latent dims must be positive; got {latentH}x{latentW}.");
        }
        if (Rect is null && Bitmap is null)
        {
            throw new HartsyInferenceException("RegionMask has neither a Rect nor a Bitmap.");
        }
        Tensor pixelMask = new Tensor(new TensorShape(1, 1, Height, Width), DType.F32);
        float* pm = (float*)pixelMask.DataPointer;
        if (Rect is RectMask rect)
        {
            int xEnd = Math.Min(Width, rect.X + rect.Width);
            int yEnd = Math.Min(Height, rect.Y + rect.Height);
            int xStart = Math.Max(0, rect.X);
            int yStart = Math.Max(0, rect.Y);
            for (int y = yStart; y < yEnd; y++)
            {
                int rowOff = y * Width;
                for (int x = xStart; x < xEnd; x++)
                {
                    pm[rowOff + x] = 1f;
                }
            }
        }
        else
        {
            float[] bmp = Bitmap!;
            for (int i = 0; i < bmp.Length; i++)
            {
                pm[i] = bmp[i];
            }
        }
        Tensor latentMask;
        if (Height % latentH == 0 && Width % latentW == 0)
        {
            latentMask = MaskBlendUtilities.DownsampleMaskAreaAverage(pixelMask, latentH, latentW);
        }
        else
        {
            latentMask = AreaResample(pixelMask, latentH, latentW);
        }
        pixelMask.Dispose();
        return latentMask;
    }

    private static unsafe Tensor AreaResample(Tensor src, int dstH, int dstW)
    {
        int srcH = (int)src.Shape[2];
        int srcW = (int)src.Shape[3];
        Tensor dst = new Tensor(new TensorShape(1, 1, dstH, dstW), DType.F32);
        float* sp = (float*)src.DataPointer;
        float* dp = (float*)dst.DataPointer;
        float scaleY = (float)srcH / dstH;
        float scaleX = (float)srcW / dstW;
        for (int dy = 0; dy < dstH; dy++)
        {
            float sy0 = dy * scaleY;
            float sy1 = (dy + 1) * scaleY;
            int y0 = (int)MathF.Floor(sy0);
            int y1 = (int)MathF.Ceiling(sy1);
            for (int dx = 0; dx < dstW; dx++)
            {
                float sx0 = dx * scaleX;
                float sx1 = (dx + 1) * scaleX;
                int x0 = (int)MathF.Floor(sx0);
                int x1 = (int)MathF.Ceiling(sx1);
                float acc = 0f;
                float area = 0f;
                for (int sy = y0; sy < y1 && sy < srcH; sy++)
                {
                    float wy = MathF.Min(sy + 1, sy1) - MathF.Max(sy, sy0);
                    if (wy <= 0f)
                    {
                        continue;
                    }
                    int rowOff = sy * srcW;
                    for (int sx = x0; sx < x1 && sx < srcW; sx++)
                    {
                        float wx = MathF.Min(sx + 1, sx1) - MathF.Max(sx, sx0);
                        if (wx <= 0f)
                        {
                            continue;
                        }
                        float w = wy * wx;
                        acc += sp[rowOff + sx] * w;
                        area += w;
                    }
                }
                dp[dy * dstW + dx] = area > 0f ? acc / area : 0f;
            }
        }
        return dst;
    }
}
