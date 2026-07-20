using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Requests;
using HartsyInference.Vision.Segmentation;
using HartsyInference.Vision.Segmentation.Sam2;

namespace HartsyInference.Engine.Vision;

/// <summary>Refines a detector's bounding box into a pixel-accurate mask with SAM 2 (box prompt). Always optional:
/// when no checkpoint is installed or the refine fails, the caller falls back to rasterizing the box.</summary>
public sealed class Sam2MaskRefiner : IDisposable
{
    // HF Sam2ImageProcessor: resize to exactly 1024×1024, then ImageNet normalize. Prompts share that square frame.
    private const int InputSize = 1024;

    private readonly Dictionary<string, SamPipeline> _cache = new(StringComparer.Ordinal);
    private readonly object _lock = new();
    private IBackend? _boundBackend;

    /// <summary>Refines the pixel-space box into a source-resolution L8 mask, or null when SAM 2 is unavailable.</summary>
    public byte[]? TryRefine(IBackend backend, string? checkpointPath, ImageData image, float boxX1, float boxY1, float boxX2, float boxY2)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (string.IsNullOrEmpty(checkpointPath))
        {
            return null;
        }
        try
        {
            SamPipeline pipeline = GetOrLoad(backend, checkpointPath);
            byte[] resized = FeatureImaging.ResizeRgb24(image, InputSize, InputSize);
            Tensor pixels = VisionTensors.ImageNetNormalized(resized, InputSize, InputSize);
            SamMaskResult best;
            try
            {
                float sx = (float)InputSize / image.Width;
                float sy = (float)InputSize / image.Height;
                SamBoxPrompt box = new SamBoxPrompt(
                    Math.Clamp(boxX1 * sx, 0f, InputSize - 1f), Math.Clamp(boxY1 * sy, 0f, InputSize - 1f),
                    Math.Clamp(boxX2 * sx, 0f, InputSize - 1f), Math.Clamp(boxY2 * sy, 0f, InputSize - 1f));
                SamMaskResult[] results = pipeline.Segment(new SamPrompt { Box = box }, pixels, InputSize, InputSize);
                if (results is null || results.Length == 0)
                {
                    return null;
                }
                best = results[0];
            }
            finally
            {
                pixels.Dispose();
            }
            return Downscale(best, image.Width, image.Height);
        }
        catch (Exception ex)
        {
            Logs.Error("[Vision] SAM 2 refine failed; falling back to the bounding-box mask.", ex);
            return null;
        }
    }

    /// <summary>Nearest-neighbour downscale of the model-resolution binary mask to the source image size.</summary>
    private static byte[] Downscale(SamMaskResult result, int width, int height)
    {
        byte[] mask = new byte[(long)width * height];
        byte[] src = result.Mask;
        for (int y = 0; y < height; y++)
        {
            int sy = Math.Clamp((int)((y + 0.5f) * result.Height / height), 0, result.Height - 1);
            int rowSrc = sy * result.Width;
            int rowDst = y * width;
            for (int x = 0; x < width; x++)
            {
                int sx = Math.Clamp((int)((x + 0.5f) * result.Width / width), 0, result.Width - 1);
                mask[rowDst + x] = src[rowSrc + sx] != 0 ? (byte)255 : (byte)0;
            }
        }
        return mask;
    }

    private SamPipeline GetOrLoad(IBackend backend, string checkpointPath)
    {
        lock (_lock)
        {
            if (!ReferenceEquals(_boundBackend, backend))
            {
                DisposeAll();
                _boundBackend = backend;
            }
            if (_cache.TryGetValue(checkpointPath, out SamPipeline? cached))
            {
                return cached;
            }
            Logs.Info($"[Vision] Loading SAM 2: {checkpointPath}");
            SamPipeline pipeline = SamPipeline.Load(backend, InferConfig(checkpointPath), checkpointPath);
            _cache[checkpointPath] = pipeline;
            return pipeline;
        }
    }

    /// <summary>Infers the SAM 2 Hiera variant from the checkpoint filename; unknown names get base+.</summary>
    private static Sam2Config InferConfig(string path)
    {
        string n = Path.GetFileNameWithoutExtension(path).ToLowerInvariant();
        if (n.Contains("tiny") || n.Contains("hiera_t"))
        {
            return Sam2Config.HieraTiny;
        }
        if (n.Contains("small") || n.Contains("hiera_s"))
        {
            return Sam2Config.HieraSmall;
        }
        if (n.Contains("large") || n.Contains("hiera_l"))
        {
            return Sam2Config.HieraLarge;
        }
        return Sam2Config.HieraBasePlus;
    }

    private void DisposeAll()
    {
        foreach (SamPipeline pipeline in _cache.Values)
        {
            try
            {
                pipeline.Dispose();
            }
            catch (Exception ex)
            {
                Logs.Error("[Vision] SAM 2 pipeline dispose failed.", ex);
            }
        }
        _cache.Clear();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_lock)
        {
            DisposeAll();
            _boundBackend = null;
        }
    }
}
