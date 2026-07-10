using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>Preprocesses RGB images for the Qwen2.5-VL vision tower, mirroring ComfyUI's
/// <c>process_qwen2vl_images</c> (comfy/text_encoders/qwen_vl.py): round both sides to multiples of
/// <c>patch · merge</c> (= 28), rescale into the <c>[minPixels, maxPixels]</c> budget, bilinear-resize
/// (align-corners = false, matching torch <c>F.interpolate</c>), then normalize with the CLIP mean/std.
/// <para>Output: <c>pixelValues [3, h_bar, w_bar]</c> F32 normalized plus the patch grid <c>(h_bar/14, w_bar/14)</c> —
/// the encoder (<see cref="Qwen25VlVisionEncoder"/>) patchifies itself, so no patch flattening happens here.</para></summary>
public sealed unsafe class Qwen25VlImageProcessor
{
    /// <summary>ComfyUI / HF Qwen2VLImageProcessor default minimum pixel budget (= 4 patches · 28²).</summary>
    public const int DefaultMinPixels = 3136;

    /// <summary>ComfyUI / HF Qwen2VLImageProcessor default maximum pixel budget (≈ 16384 patches · 28²).</summary>
    public const int DefaultMaxPixels = 12845056;

    private readonly Qwen25VlVisionConfig _config;
    private readonly int _minPixels;
    private readonly int _maxPixels;

    /// <summary>Creates an image processor. <paramref name="minPixels"/>/<paramref name="maxPixels"/> bound the
    /// resized pixel count (ComfyUI defaults 3136 / 12845056).</summary>
    public Qwen25VlImageProcessor(Qwen25VlVisionConfig config, int minPixels = DefaultMinPixels, int maxPixels = DefaultMaxPixels)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _minPixels = minPixels;
        _maxPixels = maxPixels;
    }

    /// <summary>Resizes (height, width) to multiples of <c>patch·merge</c> within the pixel budget, exactly matching
    /// ComfyUI's <c>process_qwen2vl_images</c> rounding (Python <c>round</c> = banker's, as is <see cref="Math.Round(double)"/>).</summary>
    public (int height, int width) SmartResize(int height, int width)
    {
        int factor = _config.PatchSize * _config.SpatialMergeSize;   // 28
        int hBar = (int)Math.Round(height / (double)factor) * factor;
        int wBar = (int)Math.Round(width / (double)factor) * factor;
        if ((long)hBar * wBar > _maxPixels)
        {
            double beta = Math.Sqrt((double)height * width / _maxPixels);
            hBar = Math.Max(factor, (int)Math.Floor(height / beta / factor) * factor);
            wBar = Math.Max(factor, (int)Math.Floor(width / beta / factor) * factor);
        }
        else if ((long)hBar * wBar < _minPixels)
        {
            double beta = Math.Sqrt((double)_minPixels / ((double)height * width));
            hBar = (int)Math.Ceiling(height * beta / factor) * factor;
            wBar = (int)Math.Ceiling(width * beta / factor) * factor;
        }
        return (hBar, wBar);
    }

    /// <summary>Computes the post-resize patch grid <c>(h_bar/patch, w_bar/patch)</c> from the image dims alone —
    /// same rounding math as <see cref="Preprocess"/>, no resize/normalize work. Used to count image-pad placeholder
    /// tokens before encoding.</summary>
    public (int gridH, int gridW) ComputeGrid(Tensor image)
    {
        (int srcH, int srcW) = SourceDims(image);
        (int dstH, int dstW) = SmartResize(srcH, srcW);
        return (dstH / _config.PatchSize, dstW / _config.PatchSize);
    }

    /// <summary>Preprocesses one RGB image tensor <c>[3, H, W]</c> or <c>[1, 3, H, W]</c> (F32, values in <c>[0, 1]</c> —
    /// same contract as <see cref="Qwen3VlImageProcessor"/>; ComfyUI feeds 0–1 images too).</summary>
    /// <returns><c>pixelValues [3, h_bar, w_bar]</c> F32 CLIP-normalized, and the patch grid <c>(h_bar/14, w_bar/14)</c>.</returns>
    public (Tensor pixelValues, int gridH, int gridW) Preprocess(Tensor image)
    {
        (int srcH, int srcW) = SourceDims(image);
        (int dstH, int dstW) = SmartResize(srcH, srcW);

        Tensor pixelValues = new Tensor(new TensorShape(3, dstH, dstW), DType.F32);
        ResizeAndNormalize(image, pixelValues, srcH, srcW, dstH, dstW);
        return (pixelValues, dstH / _config.PatchSize, dstW / _config.PatchSize);
    }

    private static (int height, int width) SourceDims(Tensor image)
    {
        if (image.Shape.Rank == 4 && image.Shape[0] == 1 && image.Shape[1] == 3)
            return ((int)image.Shape[2], (int)image.Shape[3]);
        if (image.Shape.Rank == 3 && image.Shape[0] == 3)
            return ((int)image.Shape[1], (int)image.Shape[2]);
        throw new ArgumentException($"Expected RGB [3, H, W] or [1, 3, H, W], got {image.Shape}.", nameof(image));
    }

    /// <summary>Bilinear resize (align-corners = false, half-pixel centers — torch <c>F.interpolate</c> bilinear) +
    /// CLIP normalization, fused per output pixel (normalizing after resize is distributive over the interpolation).</summary>
    private void ResizeAndNormalize(Tensor image, Tensor output, int srcH, int srcW, int dstH, int dstW)
    {
        float* src = (float*)image.DataPointer;   // [1,]3,H,W — a leading batch dim of 1 shares the same layout
        float* dst = (float*)output.DataPointer;
        double scaleH = (double)srcH / dstH;
        double scaleW = (double)srcW / dstW;
        for (int c = 0; c < 3; c++)
        {
            float mean = _config.ImageMean[c], std = _config.ImageStd[c];
            long srcPlane = (long)c * srcH * srcW;
            long dstPlane = (long)c * dstH * dstW;
            for (int y = 0; y < dstH; y++)
            {
                double sy = (y + 0.5) * scaleH - 0.5;
                int y0 = (int)Math.Floor(sy);
                double fy = sy - y0;
                int y0c = Math.Clamp(y0, 0, srcH - 1);
                int y1c = Math.Clamp(y0 + 1, 0, srcH - 1);
                for (int x = 0; x < dstW; x++)
                {
                    double sx = (x + 0.5) * scaleW - 0.5;
                    int x0 = (int)Math.Floor(sx);
                    double fx = sx - x0;
                    int x0c = Math.Clamp(x0, 0, srcW - 1);
                    int x1c = Math.Clamp(x0 + 1, 0, srcW - 1);

                    float v00 = src[srcPlane + (long)y0c * srcW + x0c];
                    float v01 = src[srcPlane + (long)y0c * srcW + x1c];
                    float v10 = src[srcPlane + (long)y1c * srcW + x0c];
                    float v11 = src[srcPlane + (long)y1c * srcW + x1c];
                    double top = v00 + (v01 - v00) * fx;
                    double bot = v10 + (v11 - v10) * fx;
                    float val = (float)(top + (bot - top) * fy);
                    dst[dstPlane + (long)y * dstW + x] = (val - mean) / std;
                }
            }
        }
    }
}
