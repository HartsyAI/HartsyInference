using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Multimodal;

/// <summary>LLaVA-NeXT/1.6 "anyres" image tiling: picks the best-fit resolution from a fixed set of grid
/// pinpoints, resizes+pads the image into it, and splits it into <c>tileSize×tileSize</c> tiles — plus a
/// separately-resized base/overview tile of the whole image. Ports HF's
/// <c>image_processing_llava_next.py</c> (<c>select_best_resolution</c> / <c>get_patch_output_size</c> /
/// <c>_get_padding_size</c> / <c>divide_to_patches</c>) exactly, resolved against the real installed
/// <c>transformers</c> source rather than from memory — its own docstrings disagree with each other on whether
/// image sizes are <c>(height, width)</c> or <c>(width, height)</c> (a known upstream inconsistency); this port
/// follows <c>select_best_resolution</c>'s actual body, which is unambiguously <c>(height, width)</c>
/// throughout, confirmed against <c>tests/python-reference/dump_llavanext_vision_ref.py</c>.
///
/// <para>Uses this repo's own bilinear resize kernel (matching <see cref="VlmImagePreprocessor"/>'s convention)
/// rather than HF's bicubic — the same simplification already accepted, unvalidated pixel-for-pixel, for
/// LLaVA-1.5's base-image resize. Padding is applied in raw <c>[0,255]</c> pixel space with a black (0) fill
/// BEFORE normalization (matching HF's own pipeline order), so padded regions end up at <c>-mean/std</c> per
/// channel after normalize, not exactly 0.</para></summary>
public static unsafe class LlavaNextImagePreprocessor
{
    /// <summary>Converts raw interleaved RGB (<c>H*W*3</c> bytes, row-major HWC) into a <c>[1, 3, height, width]</c>
    /// tensor of raw <c>[0,255]</c> floats (CHW, un-resized, un-normalized) — the native-resolution input
    /// <see cref="LlavaNextEncoder.Encode"/> expects, since the unpad step needs the original aspect ratio.</summary>
    public static Tensor RawToNativeTensor(ReadOnlySpan<byte> rgb, int width, int height)
    {
        if (rgb.Length < (long)width * height * 3)
            throw new ArgumentException($"rgb has {rgb.Length} bytes, expected {(long)width * height * 3} for {width}x{height}.", nameof(rgb));
        Tensor outp = new(new TensorShape(1, 3, height, width), DType.F32);
        float* dst = (float*)outp.DataPointer;
        fixed (byte* src = rgb)
        {
            for (int c = 0; c < 3; c++)
            {
                float* d = dst + (long)c * height * width;
                for (long i = 0; i < (long)height * width; i++) d[i] = src[i * 3 + c];
            }
        }
        return outp;
    }

    /// <summary>Selects the grid pinpoint that best fits <paramref name="originalSize"/>: maximizes effective
    /// (non-wasted) resolution, tie-broken by minimum wasted area. Direct port of HF's
    /// <c>image_processing_utils.select_best_resolution</c> — both the input and every candidate are
    /// <c>(height, width)</c>, and so is the return value.</summary>
    public static (int h, int w) SelectBestResolution((int h, int w) originalSize, (int h, int w)[] possibleResolutions)
    {
        int originalHeight = originalSize.h, originalWidth = originalSize.w;
        (int h, int w) bestFit = possibleResolutions[0];
        long maxEffectiveResolution = 0;
        long minWastedResolution = long.MaxValue;
        foreach ((int height, int width) in possibleResolutions)
        {
            double scale = Math.Min((double)width / originalWidth, (double)height / originalHeight);
            long downscaledWidth = (long)(originalWidth * scale), downscaledHeight = (long)(originalHeight * scale);
            long effectiveResolution = Math.Min(downscaledWidth * downscaledHeight, (long)originalWidth * originalHeight);
            long wastedResolution = (long)width * height - effectiveResolution;
            if (effectiveResolution > maxEffectiveResolution ||
                (effectiveResolution == maxEffectiveResolution && wastedResolution < minWastedResolution))
            {
                maxEffectiveResolution = effectiveResolution;
                minWastedResolution = wastedResolution;
                bestFit = (height, width);
            }
        }
        return bestFit;
    }

    /// <summary>Aspect-preserving resize-to-fit dimensions of <paramref name="original"/> inside
    /// <paramref name="target"/>. Direct port of HF's <c>image_processing_utils.get_patch_output_size</c>.</summary>
    public static (int h, int w) GetPatchOutputSize((int h, int w) original, (int h, int w) target)
    {
        double scaleW = (double)target.w / original.w, scaleH = (double)target.h / original.h;
        return scaleW < scaleH
            ? (Math.Min((int)Math.Ceiling(original.h * scaleW), target.h), target.w)
            : (target.h, Math.Min((int)Math.Ceiling(original.w * scaleH), target.w));
    }

    /// <summary>Full anyres tiling pipeline: base/overview tile (whole image, distorted-aspect resize to
    /// <paramref name="tileSize"/>²) followed by the best-fit grid's tiles in raster (row-major) order — matching
    /// HF's <c>_get_image_patches</c> (<c>[resized_original] + divide_to_patches(pad(resize(image)))</c>).
    /// <paramref name="nativeChw255"/> is the un-resized <c>[1,3,origH,origW]</c> tensor from
    /// <see cref="RawToNativeTensor"/>; each returned tile is normalized and ready for
    /// <see cref="SiglipVlmEncoder.Encode"/>.</summary>
    public static (Tensor[] tiles, int origH, int origW) Tile(
        Tensor nativeChw255, (int h, int w)[] gridPinpoints, int tileSize, float[] mean, float[] std)
    {
        int origH = (int)nativeChw255.Shape[2], origW = (int)nativeChw255.Shape[3];
        float* src = (float*)nativeChw255.DataPointer;
        List<Tensor> tiles = [];

        // Base/overview tile: resize the whole original image directly to tileSize x tileSize (aspect distorted).
        float[] baseResized = new float[3 * tileSize * tileSize];
        fixed (float* bp = baseResized)
        {
            ResizeBilinearChw(src, 3, origH, origW, bp, tileSize, tileSize);
            tiles.Add(NormalizeToTensor(bp, tileSize, mean, std));
        }

        (int bestH, int bestW) = SelectBestResolution((origH, origW), gridPinpoints);
        (int newH, int newW) = GetPatchOutputSize((origH, origW), (bestH, bestW));

        float[] resized = new float[3 * newH * newW];
        fixed (float* rp = resized) ResizeBilinearChw(src, 3, origH, origW, rp, newH, newW);

        float[] padded = new float[3 * bestH * bestW];
        fixed (float* rp = resized)
        fixed (float* pp = padded)
            PadCenterChw(rp, 3, newH, newW, pp, bestH, bestW, 0f);   // black pad, raw [0,255] space, matches HF's pipeline order

        int gh = bestH / tileSize, gw = bestW / tileSize;
        for (int i = 0; i < gh; i++)
            for (int j = 0; j < gw; j++)
            {
                float[] crop = new float[3 * tileSize * tileSize];
                fixed (float* pp = padded)
                fixed (float* cp = crop)
                {
                    CropChw(pp, 3, bestH, bestW, i * tileSize, j * tileSize, cp, tileSize, tileSize);
                    tiles.Add(NormalizeToTensor(cp, tileSize, mean, std));
                }
            }

        return (tiles.ToArray(), origH, origW);
    }

    /// <summary>Bilinear resize (center-pixel sampling, edge-clamped) — same kernel convention as
    /// <see cref="VlmImagePreprocessor"/>, generalized to non-square dimensions.</summary>
    private static void ResizeBilinearChw(float* src, int c, int srcH, int srcW, float* dst, int dstH, int dstW)
    {
        float sy = (float)srcH / dstH, sx = (float)srcW / dstW;
        for (int ch = 0; ch < c; ch++)
        {
            float* s = src + (long)ch * srcH * srcW;
            float* d = dst + (long)ch * dstH * dstW;
            for (int oy = 0; oy < dstH; oy++)
            {
                float fy = (oy + 0.5f) * sy - 0.5f;
                int y0 = (int)MathF.Floor(fy); float wy = fy - y0;
                int y0c = Math.Clamp(y0, 0, srcH - 1), y1c = Math.Clamp(y0 + 1, 0, srcH - 1);
                for (int ox = 0; ox < dstW; ox++)
                {
                    float fx = (ox + 0.5f) * sx - 0.5f;
                    int x0 = (int)MathF.Floor(fx); float wx = fx - x0;
                    int x0c = Math.Clamp(x0, 0, srcW - 1), x1c = Math.Clamp(x0 + 1, 0, srcW - 1);
                    float p00 = s[y0c * srcW + x0c], p01 = s[y0c * srcW + x1c];
                    float p10 = s[y1c * srcW + x0c], p11 = s[y1c * srcW + x1c];
                    float top = p00 + (p01 - p00) * wx, bot = p10 + (p11 - p10) * wx;
                    d[oy * dstW + ox] = top + (bot - top) * wy;
                }
            }
        }
    }

    /// <summary>Centers <paramref name="src"/> (<c>srcH×srcW</c>) inside a <c>dstH×dstW</c> canvas filled with
    /// <paramref name="padValue"/>. Matches HF's <c>_get_padding_size</c> split (floor half one side, ceil half
    /// the other).</summary>
    private static void PadCenterChw(float* src, int c, int srcH, int srcW, float* dst, int dstH, int dstW, float padValue)
    {
        int padTop = (dstH - srcH) / 2, padLeft = (dstW - srcW) / 2;
        for (int ch = 0; ch < c; ch++)
        {
            float* s = src + (long)ch * srcH * srcW;
            float* d = dst + (long)ch * dstH * dstW;
            for (int y = 0; y < dstH; y++)
                for (int x = 0; x < dstW; x++)
                {
                    int sy = y - padTop, sx = x - padLeft;
                    d[y * dstW + x] = sy >= 0 && sy < srcH && sx >= 0 && sx < srcW ? s[sy * srcW + sx] : padValue;
                }
        }
    }

    /// <summary>Extracts a <c>dstH×dstW</c> crop starting at <c>(y0, x0)</c> from a <c>srcH×srcW</c> CHW buffer.</summary>
    private static void CropChw(float* src, int c, int srcH, int srcW, int y0, int x0, float* dst, int dstH, int dstW)
    {
        for (int ch = 0; ch < c; ch++)
        {
            float* s = src + (long)ch * srcH * srcW;
            float* d = dst + (long)ch * dstH * dstW;
            for (int y = 0; y < dstH; y++)
                Buffer.MemoryCopy(s + (long)(y0 + y) * srcW + x0, d + (long)y * dstW, (long)dstW * 4, (long)dstW * 4);
        }
    }

    private static Tensor NormalizeToTensor(float* chw255, int size, float[] mean, float[] std)
    {
        Tensor t = new(new TensorShape(1, 3, size, size), DType.F32);
        float* d = (float*)t.DataPointer;
        long plane = (long)size * size;
        for (int c = 0; c < 3; c++)
        {
            float m = mean[c], s = std[c];
            long off = (long)c * plane;
            for (long i = 0; i < plane; i++) d[off + i] = (chw255[off + i] / 255f - m) / s;
        }
        return t;
    }
}
