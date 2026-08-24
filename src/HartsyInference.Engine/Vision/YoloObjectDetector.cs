using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Requests;
using HartsyInference.Vision.Detection;

namespace HartsyInference.Engine.Vision;

/// <summary>Class-prompted detection with the pure-C# YOLO v8/v11 models. The architecture variant is inferred from the checkpoint filename (the engine loads safetensors, never Ultralytics <c>.pt</c>).</summary>
public sealed class YoloObjectDetector : IDisposable
{
    private readonly Dictionary<string, YoloPipeline> _cache = new(StringComparer.Ordinal);
    private readonly object _lock = new();
    private IBackend? _boundBackend;

    /// <summary>Detects objects in <paramref name="image"/> with the checkpoint at <paramref name="checkpointPath"/>, returning pixel-space boxes above the threshold.</summary>
    public IReadOnlyList<Detection> Detect(IBackend backend, string checkpointPath, ImageData image, float threshold)
    {
        ArgumentNullException.ThrowIfNull(image);
        YoloPipeline pipeline = GetOrLoad(backend, checkpointPath);
        IReadOnlyList<YoloDetection> raw = pipeline.Detect(image.Rgb, image.Width, image.Height, confidenceThreshold: threshold);
        List<Detection> results = new List<Detection>(raw.Count);
        foreach (YoloDetection d in raw)
        {
            results.Add(new Detection
            {
                Label = pipeline.GetLabel(d.ClassId),
                Score = d.Confidence,
                X = (int)MathF.Round(d.X1),
                Y = (int)MathF.Round(d.Y1),
                Width = (int)MathF.Round(d.Width),
                Height = (int)MathF.Round(d.Height),
            });
        }
        return results;
    }

    private YoloPipeline GetOrLoad(IBackend backend, string checkpointPath)
    {
        lock (_lock)
        {
            if (!ReferenceEquals(_boundBackend, backend))
            {
                DisposeAll();
                _boundBackend = backend;
            }
            if (_cache.TryGetValue(checkpointPath, out YoloPipeline? cached))
            {
                return cached;
            }
            string name = Path.GetFileNameWithoutExtension(checkpointPath);
            bool isV11 = name.Contains("yolo11", StringComparison.OrdinalIgnoreCase) || name.Contains("yolov11", StringComparison.OrdinalIgnoreCase);
            YoloConfig config = InferConfig(name);
            Logs.Info($"[Vision] Loading YOLO ({config.Name}): {checkpointPath}");
            YoloPipeline pipeline = isV11 ? YoloPipeline.LoadV11(backend, config, checkpointPath)
                : new YoloPipeline(backend, config, checkpointPath);
            _cache[checkpointPath] = pipeline;
            return pipeline;
        }
    }

    /// <summary>Infers the YOLO backbone width from the filename; defaults to the medium variant.</summary>
    private static YoloConfig InferConfig(string name)
    {
        string n = name.ToLowerInvariant();
        bool v11 = n.Contains("yolo11") || n.Contains("yolov11");
        char size = 'm';
        foreach (char c in new[] { 'n', 's', 'm', 'l', 'x' })
        {
            if (n.Contains($"yolov8{c}") || n.Contains($"yolo11{c}") || n.Contains($"yolov11{c}"))
            {
                size = c;
                break;
            }
        }
        return (v11, size) switch
        {
            (true, 'n') => YoloConfig.YoloV11n,
            (true, 's') => YoloConfig.YoloV11s,
            (true, 'l') => YoloConfig.YoloV11l,
            (true, 'x') => YoloConfig.YoloV11x,
            (true, _) => YoloConfig.YoloV11m,
            (false, 'n') => YoloConfig.YoloV8n,
            (false, 's') => YoloConfig.YoloV8s,
            (false, 'l') => YoloConfig.YoloV8l,
            (false, 'x') => YoloConfig.YoloV8x,
            _ => YoloConfig.YoloV8m,
        };
    }

    private void DisposeAll()
    {
        foreach (YoloPipeline pipeline in _cache.Values)
        {
            try
            {
                pipeline.Dispose();
            }
            catch (Exception ex)
            {
                Logs.Error("[Vision] YOLO pipeline dispose failed.", ex);
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
