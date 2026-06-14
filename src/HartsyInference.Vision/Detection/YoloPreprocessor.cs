using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>YOLO image preprocessing — Ultralytics-compatible letterbox resize + 0..1 normalize +
/// HWC→CHW transpose. The output tensor is ready to feed into a YOLO detection backbone.
/// <para><b>Letterbox vs. naive resize.</b> YOLO does <i>not</i> stretch the image to a square.
/// Instead it scales by the longer-edge ratio, pads the short edge with gray (114,114,114), and
/// aligns padding to the model's stride (32 by default). This preserves aspect ratio so detections
/// can be mapped back to original-image coordinates after NMS via <see cref="Transform.Invert"/>.</para>
/// <para><b>RGB vs. BGR.</b> Ultralytics decodes images with OpenCV (BGR) and converts to RGB
/// before the model. Our entry point accepts HWC-packed RGB u8 already (the same convention as
/// <see cref="Clip.ClipImagePreprocessor"/>) — callers using BGR-native decoders are responsible
/// for swapping channels before passing pixels in.</para>
/// <para><b>Normalization.</b> Just <c>/255.0</c> — no ImageNet mean/std subtraction. Differs from
/// CLIP and most classifiers; using ImageNet mean/std here silently degrades mAP.</para></summary>
public sealed unsafe class YoloPreprocessor
{
    /// <summary>The gray padding color Ultralytics uses for letterbox.</summary>
    public const byte PadValue = 114;

    private readonly int _targetSize;
    private readonly int _stride;
    private readonly bool _stridePadAlign;

    /// <summary>Target side length in pixels (640 standard, 320/416/512/1280 also common).</summary>
    public int TargetSize => _targetSize;

    /// <summary>Model stride (max downsample factor). 32 for YOLOv8/YOLO11.</summary>
    public int Stride => _stride;

    /// <summary>Creates a preprocessor. <paramref name="stridePadAlign"/> controls whether padding
    /// is rounded to the nearest stride multiple (Ultralytics <c>auto=True</c>, the default) — this
    /// produces non-square letterbox dimensions for non-square inputs (e.g. 480×640 instead of
    /// 640×640), saving compute. Set to false to always emit exact <c>targetSize × targetSize</c>
    /// tensors (matches <c>auto=False</c>).</summary>
    public YoloPreprocessor(int targetSize = 640, int stride = 32, bool stridePadAlign = false)
    {
        if (targetSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(targetSize), targetSize, "Target size must be positive.");
        if (stride <= 0 || targetSize % stride != 0)
            throw new ArgumentException($"Target size {targetSize} must be a positive multiple of stride {stride}.", nameof(stride));

        _targetSize = targetSize;
        _stride = stride;
        _stridePadAlign = stridePadAlign;
    }

    /// <summary>Runs the full letterbox + normalize + CHW transpose pipeline on one image.
    /// Returns the prepared tensor and a <see cref="Transform"/> describing how the source
    /// coordinates map into the letterbox canvas — needed by <see cref="Transform.Invert"/>
    /// to project detected boxes back into the original-image coordinate frame.</summary>
    /// <param name="rgbPixels">HWC-packed RGB u8, length <c>srcWidth * srcHeight * 3</c>.</param>
    /// <param name="srcWidth">Source width in pixels.</param>
    /// <param name="srcHeight">Source height in pixels.</param>
    public (Tensor input, Transform transform) Preprocess(ReadOnlySpan<byte> rgbPixels, int srcWidth, int srcHeight)
    {
        if (srcWidth <= 0 || srcHeight <= 0)
            throw new ArgumentException($"Source dimensions must be positive; got {srcWidth}x{srcHeight}.");
        int expected = srcWidth * srcHeight * 3;
        if (rgbPixels.Length != expected)
            throw new ArgumentException($"rgbPixels length {rgbPixels.Length} != expected {expected} for {srcWidth}x{srcHeight} RGB.", nameof(rgbPixels));

        Transform transform = ComputeTransform(srcWidth, srcHeight);

        Tensor output = new Tensor(new TensorShape(1, 3, transform.PaddedHeight, transform.PaddedWidth), DType.F32);
        FillPad(output, transform.PaddedWidth, transform.PaddedHeight);
        BilinearResizeAndWrite(rgbPixels, srcWidth, srcHeight, output, transform);
        return (output, transform);
    }

    /// <summary>Computes the letterbox transform — scale ratio, resized dims, padding offsets,
    /// and final canvas size — without performing the resize. Useful for batch planning or to
    /// re-use the same transform across multiple frames of the same resolution.</summary>
    public Transform ComputeTransform(int srcWidth, int srcHeight)
    {
        float r = MathF.Min((float)_targetSize / srcHeight, (float)_targetSize / srcWidth);
        int newW = (int)MathF.Round(srcWidth * r);
        int newH = (int)MathF.Round(srcHeight * r);

        int dw = _targetSize - newW;
        int dh = _targetSize - newH;

        if (_stridePadAlign)
        {
            dw %= _stride;
            dh %= _stride;
        }

        int top = dh / 2;
        int left = dw / 2;
        int paddedH = newH + dh;
        int paddedW = newW + dw;

        return new Transform(srcWidth, srcHeight, newW, newH, r, left, top, paddedW, paddedH);
    }

    private static void FillPad(Tensor t, int width, int height)
    {
        // CHW layout: channel-major. Fill all three planes with the pad value (normalized).
        float padNorm = PadValue / 255f;
        float* p = (float*)t.DataPointer;
        long plane = (long)width * height;
        long total = plane * 3;
        for (long i = 0; i < total; i++)
            p[i] = padNorm;
    }

    /// <summary>Bilinear resize of the source HWC u8 region into the inner letterbox window
    /// (<c>[top:top+newH, left:left+newW]</c>) of the CHW float canvas, dividing by 255 inline.
    /// Bilinear (not bicubic) matches Ultralytics' default <c>cv2.INTER_LINEAR</c> resize.</summary>
    private static void BilinearResizeAndWrite(ReadOnlySpan<byte> src, int srcW, int srcH, Tensor dst, Transform tr)
    {
        int dstStride = tr.PaddedWidth;
        long plane = (long)tr.PaddedWidth * tr.PaddedHeight;
        float* dstPtr = (float*)dst.DataPointer;

        float scaleX = (float)srcW / tr.ResizedWidth;
        float scaleY = (float)srcH / tr.ResizedHeight;

        for (int oy = 0; oy < tr.ResizedHeight; oy++)
        {
            float sy = (oy + 0.5f) * scaleY - 0.5f;
            int sy0 = (int)MathF.Floor(sy);
            float fy = sy - sy0;
            int y0 = Math.Clamp(sy0,     0, srcH - 1);
            int y1 = Math.Clamp(sy0 + 1, 0, srcH - 1);
            float wy1 = fy;
            float wy0 = 1f - fy;

            for (int ox = 0; ox < tr.ResizedWidth; ox++)
            {
                float sx = (ox + 0.5f) * scaleX - 0.5f;
                int sx0 = (int)MathF.Floor(sx);
                float fx = sx - sx0;
                int x0 = Math.Clamp(sx0,     0, srcW - 1);
                int x1 = Math.Clamp(sx0 + 1, 0, srcW - 1);
                float wx1 = fx;
                float wx0 = 1f - fx;

                int dstY = tr.PadTop + oy;
                int dstX = tr.PadLeft + ox;
                long dstPos = (long)dstY * dstStride + dstX;

                for (int c = 0; c < 3; c++)
                {
                    byte p00 = src[(y0 * srcW + x0) * 3 + c];
                    byte p01 = src[(y0 * srcW + x1) * 3 + c];
                    byte p10 = src[(y1 * srcW + x0) * 3 + c];
                    byte p11 = src[(y1 * srcW + x1) * 3 + c];
                    float v = (p00 * wx0 + p01 * wx1) * wy0
                            + (p10 * wx0 + p11 * wx1) * wy1;
                    dstPtr[c * plane + dstPos] = v * (1f / 255f);
                }
            }
        }
    }

    /// <summary>Letterbox transform record — captures every parameter needed to (a) describe the
    /// preprocessing for debugging and (b) un-do the mapping after NMS so detection boxes can be
    /// reported in original-image coordinates.</summary>
    public readonly record struct Transform(
        int SourceWidth,
        int SourceHeight,
        int ResizedWidth,
        int ResizedHeight,
        float Scale,
        int PadLeft,
        int PadTop,
        int PaddedWidth,
        int PaddedHeight)
    {
        /// <summary>Maps a box from letterboxed-canvas xyxy coordinates back to source-image xyxy
        /// coordinates. Used after NMS to report detections in the caller's coordinate frame.</summary>
        public (float x1, float y1, float x2, float y2) Invert(float x1, float y1, float x2, float y2)
        {
            float invScale = 1f / Scale;
            float sx1 = (x1 - PadLeft) * invScale;
            float sy1 = (y1 - PadTop)  * invScale;
            float sx2 = (x2 - PadLeft) * invScale;
            float sy2 = (y2 - PadTop)  * invScale;
            sx1 = Math.Clamp(sx1, 0f, SourceWidth);
            sy1 = Math.Clamp(sy1, 0f, SourceHeight);
            sx2 = Math.Clamp(sx2, 0f, SourceWidth);
            sy2 = Math.Clamp(sy2, 0f, SourceHeight);
            return (sx1, sy1, sx2, sy2);
        }
    }
}
