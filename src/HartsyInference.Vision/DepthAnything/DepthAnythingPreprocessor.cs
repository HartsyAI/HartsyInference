using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.Vision.DepthAnything;

/// <summary>Depth-Anything-V2 image pre/post-processing, matching the official pipeline (and the ComfyUI
/// DepthAnythingV2 preprocessor): scale so the shorter side reaches the lower bound (default 518) keeping
/// aspect ratio, snap both sides to multiples of 14, bicubic resize (OpenCV <c>INTER_CUBIC</c>, A = −0.75)
/// on /255 floats, then ImageNet mean/std normalization. Postprocessing resizes the depth map back to the
/// source resolution (bilinear, <c>align_corners=True</c>) and min-max normalizes to [0,1] — the grayscale
/// relative-depth conditioning ControlNet/Flux-Depth expects (1 = nearest). Host code by design: runs once
/// per generation, outside the per-step hot path.</summary>
public sealed unsafe class DepthAnythingPreprocessor
{
    private readonly int _lowerBound;
    private readonly int _multipleOf;
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    public DepthAnythingPreprocessor(int lowerBound = 518, int multipleOf = 14)
    {
        if (lowerBound <= 0 || multipleOf <= 0 || lowerBound % multipleOf != 0)
            throw new ArgumentException($"Lower bound {lowerBound} must be a positive multiple of {multipleOf}.");
        _lowerBound = lowerBound;
        _multipleOf = multipleOf;
    }

    /// <summary>Network input size for a source image: aspect-preserving scale so both sides ≥ the lower
    /// bound (<c>resize_method="lower_bound"</c>), each side rounded to the nearest multiple of 14 (never
    /// below the bound).</summary>
    public (int Width, int Height) ComputeTargetSize(int srcWidth, int srcHeight)
    {
        if (srcWidth <= 0 || srcHeight <= 0)
            throw new ArgumentException($"Source dimensions must be positive; got {srcWidth}x{srcHeight}.");
        float scale = Math.Max((float)_lowerBound / srcWidth, (float)_lowerBound / srcHeight);
        return (ConstrainToMultiple(scale * srcWidth), ConstrainToMultiple(scale * srcHeight));
    }

    /// <summary>Resizes + normalizes an interleaved-RGB24 image into the network input
    /// <c>[1, 3, targetH, targetW]</c> (target size from <see cref="ComputeTargetSize"/>).</summary>
    public Tensor Preprocess(ReadOnlySpan<byte> rgbPixels, int srcWidth, int srcHeight)
    {
        long expected = (long)srcWidth * srcHeight * 3;
        if (srcWidth <= 0 || srcHeight <= 0 || rgbPixels.Length != expected)
            throw new ArgumentException($"rgbPixels length {rgbPixels.Length} != {srcWidth}x{srcHeight}x3.", nameof(rgbPixels));

        (int outW, int outH) = ComputeTargetSize(srcWidth, srcHeight);
        Tensor output = new(new TensorShape(1, 3, outH, outW), DType.F32);
        float* dst = (float*)output.DataPointer;
        long plane = (long)outH * outW;
        float invScaleX = (float)srcWidth / outW;
        float invScaleY = (float)srcHeight / outH;

        Span<float> wy = stackalloc float[4];
        Span<float> wx = stackalloc float[4];
        fixed (byte* src = rgbPixels)
        {
            for (int oy = 0; oy < outH; oy++)
            {
                float sy = (oy + 0.5f) * invScaleY - 0.5f;
                int y0 = (int)MathF.Floor(sy);
                Dinov2VisionEncoder.CubicWeights(sy - y0, wy);
                for (int ox = 0; ox < outW; ox++)
                {
                    float sx = (ox + 0.5f) * invScaleX - 0.5f;
                    int x0 = (int)MathF.Floor(sx);
                    Dinov2VisionEncoder.CubicWeights(sx - x0, wx);
                    for (int c = 0; c < 3; c++)
                    {
                        float acc = 0f;
                        for (int ty = 0; ty < 4; ty++)
                        {
                            int yy = Math.Clamp(y0 - 1 + ty, 0, srcHeight - 1);
                            float row = 0f;
                            for (int tx = 0; tx < 4; tx++)
                            {
                                int xx = Math.Clamp(x0 - 1 + tx, 0, srcWidth - 1);
                                row += wx[tx] * src[((long)yy * srcWidth + xx) * 3 + c];
                            }
                            acc += wy[ty] * row;
                        }
                        dst[c * plane + (long)oy * outW + ox] = (acc / 255f - Mean[c]) / Std[c];
                    }
                }
            }
        }
        return output;
    }

    /// <summary>Resizes a network-resolution depth map <c>[1,1,h,w]</c> back to the source resolution
    /// (bilinear, <c>align_corners=True</c> like the official <c>infer_image</c>) and min-max normalizes to
    /// [0,1]. Returns a row-major <c>dstWidth·dstHeight</c> buffer; 1 = nearest.</summary>
    public static float[] PostprocessToUnit(Tensor depth, int dstWidth, int dstHeight)
        => PostprocessToUnit(depth, dstWidth, dstHeight, minMaxNormalize: true);

    /// <summary>Resize + scale variant: <paramref name="minMaxNormalize"/> true = full-range min-max stretch (the comfyui_controlnet_aux convention SD ControlNets expect); false = max-only scaling (<c>depth / max</c>, the BFL FLUX.1-Depth reference form — no min subtraction, so far regions keep their natural non-zero level instead of saturating to black; the stretched distribution made FLUX Depth hallucinate caption-bar/web-UI content on real scenes). The resize kernel follows the same split: min-max mode uses bilinear <c>align_corners=True</c> (official <c>infer_image</c>, what comfyui_controlnet_aux feeds SD ControlNets); flux mode uses antialiased bicubic <c>align_corners=False</c> (BFL's <c>DepthImageEncoder</c> — <c>F.interpolate(mode="bicubic", antialias=True)</c> — so the conditioning map carries the same edge profile FLUX.1-Depth was trained on).</summary>
    public static float[] PostprocessToUnit(Tensor depth, int dstWidth, int dstHeight, bool minMaxNormalize)
    {
        if (depth.Shape.Rank != 4 || depth.Shape[0] != 1 || depth.Shape[1] != 1)
            throw new ArgumentException($"depth must be [1,1,h,w]; got {depth.Shape}.", nameof(depth));
        int inH = (int)depth.Shape[2], inW = (int)depth.Shape[3];
        float* src = (float*)depth.DataPointer;
        float[] resized = new float[(long)dstWidth * dstHeight];

        if (minMaxNormalize)
        {
            ResizeBilinearAlignCorners(src, inW, inH, resized, dstWidth, dstHeight);
            NormalizeToUnit(resized);
        }
        else
        {
            // torch's antialias=True path uses the PIL A = −0.5 kernel (not the non-AA −0.75 kernel).
            ReadOnlySpan<float> plane = new((void*)src, inH * inW);
            Codec.Resample.BicubicPlane(plane, inW, inH, resized, dstWidth, dstHeight, a: -0.5f, antialias: true);
            float max = 0f;
            foreach (float v in resized) if (v > max) max = v;
            if (max > 0f)
            {
                float inv = 1f / max;
                for (long i = 0; i < resized.LongLength; i++) resized[i] *= inv;
            }
        }
        return resized;
    }

    /// <summary>In-place min-max normalization to [0,1]; a constant map becomes all zeros.</summary>
    public static void NormalizeToUnit(Span<float> depth)
    {
        float min = float.MaxValue, max = float.MinValue;
        foreach (float v in depth) { if (v < min) min = v; if (v > max) max = v; }
        float range = max - min;
        if (range <= 0f) { depth.Clear(); return; }
        for (int i = 0; i < depth.Length; i++) depth[i] = (depth[i] - min) / range;
    }

    /// <summary>Renders a [0,1] depth buffer as interleaved-RGB24 grayscale (255 = nearest).</summary>
    public static byte[] ToGrayscaleRgb24(ReadOnlySpan<float> unitDepth) =>
        Codec.ImageTensor.UnitGrayscaleToRgb24(unitDepth);

    /// <summary>Bilinear <c>align_corners=True</c> plane resize — the official <c>infer_image</c> depth
    /// upsample convention.</summary>
    private static void ResizeBilinearAlignCorners(float* src, int inW, int inH, float[] dst, int dstWidth, int dstHeight)
    {
        for (int oy = 0; oy < dstHeight; oy++)
        {
            float sy = dstHeight == 1 ? 0f : oy * (float)(inH - 1) / (dstHeight - 1);
            int y0 = (int)MathF.Floor(sy);
            float fy = sy - y0;
            int y0c = Math.Clamp(y0, 0, inH - 1), y1c = Math.Clamp(y0 + 1, 0, inH - 1);
            for (int ox = 0; ox < dstWidth; ox++)
            {
                float sx = dstWidth == 1 ? 0f : ox * (float)(inW - 1) / (dstWidth - 1);
                int x0 = (int)MathF.Floor(sx);
                float fx = sx - x0;
                int x0c = Math.Clamp(x0, 0, inW - 1), x1c = Math.Clamp(x0 + 1, 0, inW - 1);
                float v0 = src[y0c * inW + x0c] * (1f - fx) + src[y0c * inW + x1c] * fx;
                float v1 = src[y1c * inW + x0c] * (1f - fx) + src[y1c * inW + x1c] * fx;
                dst[(long)oy * dstWidth + ox] = v0 * (1f - fy) + v1 * fy;
            }
        }
    }

    /// <summary>Rounds to the nearest multiple of 14, bumping up when that would fall below the lower bound
    /// (the official <c>constrain_to_multiple_of</c> with <c>min_val</c>).</summary>
    private int ConstrainToMultiple(float x)
    {
        int y = (int)Math.Round(x / _multipleOf, MidpointRounding.ToEven) * _multipleOf;
        if (y < _lowerBound) y = (int)Math.Ceiling(x / _multipleOf) * _multipleOf;
        return y;
    }
}
