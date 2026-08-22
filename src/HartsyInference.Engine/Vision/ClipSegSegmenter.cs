using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Vision.Segmentation;

namespace HartsyInference.Engine.Vision;

/// <summary>Free-text segmentation with the pure-C# CLIPSeg (<c>clipseg-rd64-refined</c>): a soft [0,1] mask at 224² that is bilinearly upsampled to the source resolution and binarized at the request threshold.</summary>
public sealed class ClipSegSegmenter : IDisposable
{
    private readonly Dictionary<string, ClipSegPipeline> _cache = new(StringComparer.Ordinal);
    private readonly object _lock = new();

    /// <summary>Segments <paramref name="query"/> in <paramref name="image"/>, returning a source-resolution L8 mask (255 = matched), or null when nothing scored above <paramref name="threshold"/>.</summary>
    public byte[]? Segment(IBackend backend, string modelDirectory, ImageData image, string query, float threshold)
    {
        ArgumentNullException.ThrowIfNull(image);
        string target = (query ?? "").Trim();
        if (target.Length == 0)
        {
            throw new InvalidOperationException("CLIPSeg needs a text query; supply VisionRequest.Prompt.");
        }
        ClipSegPipeline pipeline = GetOrLoad(modelDirectory);
        Tensor pixels = FeatureImaging.RgbToTensorZeroOne(image.Rgb, image.Width, image.Height);
        float[] soft;
        int maskW;
        int maskH;
        try
        {
            soft = pipeline.Segment(backend, pixels, target, out maskW, out maskH);
        }
        catch (Exception ex)
        {
            Logs.Error($"[Vision] CLIPSeg forward failed for query '{target}'.", ex);
            throw;
        }
        finally
        {
            pixels.Dispose();
        }
        int width = image.Width;
        int height = image.Height;
        byte[] mask = new byte[(long)width * height];
        int matched = 0;
        for (int y = 0; y < height; y++)
        {
            float fy = (y + 0.5f) * maskH / height - 0.5f;
            int y0 = Math.Clamp((int)MathF.Floor(fy), 0, maskH - 1);
            int y1 = Math.Min(y0 + 1, maskH - 1);
            float wy = MathF.Max(0f, fy - y0);
            for (int x = 0; x < width; x++)
            {
                float fx = (x + 0.5f) * maskW / width - 0.5f;
                int x0 = Math.Clamp((int)MathF.Floor(fx), 0, maskW - 1);
                int x1 = Math.Min(x0 + 1, maskW - 1);
                float wx = MathF.Max(0f, fx - x0);
                float top = soft[y0 * maskW + x0] + (soft[y0 * maskW + x1] - soft[y0 * maskW + x0]) * wx;
                float bot = soft[y1 * maskW + x0] + (soft[y1 * maskW + x1] - soft[y1 * maskW + x0]) * wx;
                float v = top + (bot - top) * wy;
                if (v >= threshold)
                {
                    mask[y * width + x] = 255;
                    matched++;
                }
            }
        }
        if (matched == 0)
        {
            Logs.Info($"[Vision] CLIPSeg '{target}': nothing matched above threshold {threshold:F2}.");
            return null;
        }
        Logs.Info($"[Vision] CLIPSeg '{target}': matched {matched} px ({100.0 * matched / (width * (double)height):F1}% of image) at threshold {threshold:F2}.");
        return mask;
    }

    private ClipSegPipeline GetOrLoad(string modelDirectory)
    {
        lock (_lock)
        {
            if (_cache.TryGetValue(modelDirectory, out ClipSegPipeline? cached))
            {
                return cached;
            }
            Logs.Info($"[Vision] Loading CLIPSeg: {modelDirectory}");
            ClipSegPipeline pipeline = new ClipSegPipeline(modelDirectory);
            _cache[modelDirectory] = pipeline;
            return pipeline;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_lock)
        {
            foreach (ClipSegPipeline pipeline in _cache.Values)
            {
                try
                {
                    pipeline.Dispose();
                }
                catch (Exception ex)
                {
                    Logs.Error("[Vision] CLIPSeg pipeline dispose failed.", ex);
                }
            }
            _cache.Clear();
        }
    }
}
