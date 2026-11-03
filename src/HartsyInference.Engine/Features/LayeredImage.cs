using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Features;

/// <summary>The processing rules for layered image results (<see cref="ImageResult.Layers"/>):
/// <list type="bullet">
///   <item>The composite is always recomputed from the layers: straight-alpha "over", bottom first.</item>
///   <item>Resize applies one resampling to every layer, premultiplied then un-premultiplied, so a transparent pixel's
///         colour never bleeds into its neighbours.</item>
///   <item>Crop applies one rectangle to every layer.</item>
///   <item>Post-steps with no defined layered behaviour (inpaint recomposite and crop-to-mask, the refiner pass, segment
///         refinement, background removal) are refused up front by <see cref="Violation"/>.</item>
/// </list>
/// Consumers that resize or crop a layered result (the CLI, the API, the SwarmUI extension) use these helpers so the
/// layers and the composite stay consistent.</summary>
public static class LayeredImage
{
    /// <summary>Why <paramref name="request"/> cannot run on a family with layered output, or null when it can.</summary>
    public static string? Violation(ImageRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Inpaint is not null)
            return "Inpainting is not defined for layered output yet: the recomposite and crop-to-mask steps have no per-layer rule.";
        if (request.Refiner is not null)
            return "A refiner pass is not defined for layered output yet: it would rework the composite and leave the layers stale.";
        if (request.RemoveBackground == true)
            return "Background removal is not defined for layered output: the layers already carry their own alpha.";
        if (SegmentRefinement.HasSegmentParts(request.Prompt))
            return "Segment refinement is not defined for layered output yet: it would rework the composite and leave the layers stale.";
        return null;
    }

    /// <summary>Throws when <paramref name="result"/>'s layers are out of order or not sized for the image.</summary>
    public static void Validate(ImageResult result)
    {
        ArgumentNullException.ThrowIfNull(result);
        if (result.Layers is not { Count: > 0 } layers)
            throw new InvalidOperationException("A layered result needs at least one layer.");
        long pixels = (long)result.Width * result.Height;
        for (int i = 0; i < layers.Count; i++)
        {
            ImageLayer layer = layers[i];
            if (layer.Index != i)
                throw new InvalidOperationException($"Layer at position {i} has Index {layer.Index}; layers must be ordered back to front from 0.");
            if (layer.Width != result.Width || layer.Height != result.Height)
                throw new InvalidOperationException($"Layer {i} is {layer.Width}x{layer.Height}; the image is {result.Width}x{result.Height}.");
            if (layer.Rgb.Length != pixels * 3 || layer.Alpha.Length != pixels)
                throw new InvalidOperationException($"Layer {i} does not carry {result.Width}x{result.Height} RGB and alpha planes.");
        }
    }

    /// <summary><paramref name="result"/> with <see cref="ImageResult.Rgb"/> and <see cref="ImageResult.Alpha"/> recomputed
    /// from its layers.</summary>
    public static ImageResult WithComposite(ImageResult result)
    {
        Validate(result);
        (byte[] rgb, byte[] alpha) = Composite(result.Layers!);
        return result with { Rgb = rgb, Alpha = alpha };
    }

    /// <summary>Straight-alpha "over" of <paramref name="layers"/>, bottom first: returns straight RGB and alpha.</summary>
    public static (byte[] Rgb, byte[] Alpha) Composite(IReadOnlyList<ImageLayer> layers)
    {
        ArgumentNullException.ThrowIfNull(layers);
        if (layers.Count == 0)
            throw new ArgumentException("Nothing to composite.", nameof(layers));
        int pixels = layers[0].Width * layers[0].Height;
        // Premultiplied accumulators: over is c = c_src·a_src + c_dst·(1 − a_src), a = a_src + a_dst·(1 − a_src).
        float[] premul = new float[pixels * 3];
        float[] coverage = new float[pixels];
        foreach (ImageLayer layer in layers)
        {
            for (int p = 0; p < pixels; p++)
            {
                float a = layer.Alpha[p] / 255f;
                float keep = 1 - a;
                for (int c = 0; c < 3; c++)
                    premul[p * 3 + c] = layer.Rgb[p * 3 + c] / 255f * a + premul[p * 3 + c] * keep;
                coverage[p] = a + coverage[p] * keep;
            }
        }
        byte[] rgb = new byte[pixels * 3];
        byte[] alpha = new byte[pixels];
        for (int p = 0; p < pixels; p++)
        {
            float a = coverage[p];
            alpha[p] = ToByte(a);
            for (int c = 0; c < 3; c++)
                rgb[p * 3 + c] = a > 0 ? ToByte(premul[p * 3 + c] / a) : (byte)0;
        }
        return (rgb, alpha);
    }

    /// <summary><paramref name="result"/> with every layer resampled to <paramref name="width"/> × <paramref name="height"/>
    /// (antialiased triangle filter on premultiplied values) and the composite recomputed.</summary>
    public static ImageResult Resize(ImageResult result, int width, int height)
    {
        Validate(result);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        List<ImageLayer> resized = result.Layers!.Select(l => ResizeLayer(l, width, height)).ToList();
        return WithComposite(result with { Width = width, Height = height, Layers = resized });
    }

    /// <summary><paramref name="result"/> with every layer cut to the same rectangle and the composite recomputed.</summary>
    public static ImageResult Crop(ImageResult result, int x, int y, int width, int height)
    {
        Validate(result);
        if (x < 0 || y < 0 || width < 1 || height < 1 || x + width > result.Width || y + height > result.Height)
            throw new ArgumentException($"Crop ({x}, {y}, {width}x{height}) is outside the {result.Width}x{result.Height} image.");
        List<ImageLayer> cropped = result.Layers!.Select(l => l with
        {
            Rgb = CropPlane(l.Rgb, l.Width, 3, x, y, width, height),
            Alpha = CropPlane(l.Alpha, l.Width, 1, x, y, width, height),
            Width = width,
            Height = height,
        }).ToList();
        return WithComposite(result with { Width = width, Height = height, Layers = cropped });
    }

    private static ImageLayer ResizeLayer(ImageLayer layer, int width, int height)
    {
        int src = layer.Width * layer.Height;
        // Planes: R·a, G·a, B·a, a. Resampling premultiplied values keeps a transparent pixel's colour out of the mix.
        float[][] planes = [new float[src], new float[src], new float[src], new float[src]];
        for (int p = 0; p < src; p++)
        {
            float a = layer.Alpha[p] / 255f;
            planes[3][p] = a;
            for (int c = 0; c < 3; c++)
                planes[c][p] = layer.Rgb[p * 3 + c] / 255f * a;
        }
        float[][] outPlanes = planes.Select(pl => Resample(pl, layer.Width, layer.Height, width, height)).ToArray();
        int dst = width * height;
        byte[] rgb = new byte[dst * 3];
        byte[] alpha = new byte[dst];
        for (int p = 0; p < dst; p++)
        {
            float a = Math.Clamp(outPlanes[3][p], 0, 1);
            alpha[p] = ToByte(a);
            for (int c = 0; c < 3; c++)
                rgb[p * 3 + c] = a > 0 ? ToByte(outPlanes[c][p] / a) : (byte)0;
        }
        return layer with { Rgb = rgb, Alpha = alpha, Width = width, Height = height };
    }

    /// <summary>Separable resample with a triangle kernel whose support widens with the downscale factor (antialiased
    /// bilinear), sampling at pixel centres.</summary>
    private static float[] Resample(float[] plane, int srcW, int srcH, int dstW, int dstH)
    {
        float[] horizontal = new float[dstW * srcH];
        (int[] starts, float[][] weights) = Kernel(srcW, dstW);
        for (int y = 0; y < srcH; y++)
        {
            for (int x = 0; x < dstW; x++)
            {
                float sum = 0;
                for (int k = 0; k < weights[x].Length; k++)
                    sum += plane[y * srcW + starts[x] + k] * weights[x][k];
                horizontal[y * dstW + x] = sum;
            }
        }
        float[] output = new float[dstW * dstH];
        (starts, weights) = Kernel(srcH, dstH);
        for (int y = 0; y < dstH; y++)
        {
            for (int x = 0; x < dstW; x++)
            {
                float sum = 0;
                for (int k = 0; k < weights[y].Length; k++)
                    sum += horizontal[(starts[y] + k) * dstW + x] * weights[y][k];
                output[y * dstW + x] = sum;
            }
        }
        return output;
    }

    private static (int[] Starts, float[][] Weights) Kernel(int src, int dst)
    {
        double scale = (double)src / dst;
        double support = Math.Max(1.0, scale);
        int[] starts = new int[dst];
        float[][] weights = new float[dst][];
        for (int i = 0; i < dst; i++)
        {
            double centre = (i + 0.5) * scale;
            int lo = Math.Max(0, (int)Math.Floor(centre - support));
            int hi = Math.Min(src - 1, (int)Math.Ceiling(centre + support));
            double[] w = new double[hi - lo + 1];
            double total = 0;
            for (int j = lo; j <= hi; j++)
            {
                double d = Math.Abs(j + 0.5 - centre) / support;
                w[j - lo] = Math.Max(0, 1 - d);
                total += w[j - lo];
            }
            starts[i] = lo;
            weights[i] = w.Select(v => (float)(total > 0 ? v / total : 0)).ToArray();
        }
        return (starts, weights);
    }

    private static byte[] CropPlane(byte[] plane, int width, int channels, int x, int y, int w, int h)
    {
        byte[] output = new byte[w * h * channels];
        for (int row = 0; row < h; row++)
            Array.Copy(plane, ((y + row) * width + x) * channels, output, row * w * channels, w * channels);
        return output;
    }

    private static byte ToByte(float unit) => (byte)Math.Clamp((int)MathF.Round(unit * 255f), 0, 255);
}
