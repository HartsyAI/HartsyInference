using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Pipelines;
using HartsyInference.Engine.Requests;
using HartsyInference.Vision.Detection;

namespace HartsyInference.Engine.Vision;

/// <summary>Closed-set (80-class COCO) detection with the pure-C# RT-DETR (<c>rtdetr_r18vd</c>) — a transformer, NMS-free alternative to YOLO. Pipelines are cached per checkpoint because weight load + conversion is expensive.</summary>
public sealed class RtDetrObjectDetector : IDisposable
{
    private readonly Dictionary<string, RtDetrPipeline> _cache = new(StringComparer.Ordinal);
    private readonly object _lock = new();
    private IBackend? _boundBackend;

    /// <summary>Detects COCO objects in <paramref name="image"/>, returning pixel-space boxes above the threshold.</summary>
    public IReadOnlyList<Detection> Detect(IBackend backend, string checkpointPath, ImageData image, float threshold)
    {
        ArgumentNullException.ThrowIfNull(image);
        RtDetrPipeline pipeline = GetOrLoad(backend, checkpointPath);
        IReadOnlyList<DetectionResult> raw = pipeline.Detect(image.Rgb, image.Width, image.Height, threshold);
        List<Detection> results = new List<Detection>(raw.Count);
        foreach (DetectionResult d in raw)
        {
            results.Add(new Detection
            {
                Label = d.Label,
                Score = d.Confidence,
                X = (int)MathF.Round(d.X * image.Width),
                Y = (int)MathF.Round(d.Y * image.Height),
                Width = (int)MathF.Round(d.Width * image.Width),
                Height = (int)MathF.Round(d.Height * image.Height),
            });
        }
        return results;
    }

    private RtDetrPipeline GetOrLoad(IBackend backend, string checkpointPath)
    {
        lock (_lock)
        {
            if (!ReferenceEquals(_boundBackend, backend))
            {
                DisposeAll();
                _boundBackend = backend;
            }
            if (_cache.TryGetValue(checkpointPath, out RtDetrPipeline? cached))
            {
                return cached;
            }
            Logs.Info($"[Vision] Loading RT-DETR (r18vd): {checkpointPath}");
            RtDetrPipeline pipeline = new RtDetrPipeline(backend, RtDetrConfig.R18vd, checkpointPath, inputSize: 640, labels: CocoLabels.Names);
            _cache[checkpointPath] = pipeline;
            return pipeline;
        }
    }

    private void DisposeAll()
    {
        foreach (RtDetrPipeline pipeline in _cache.Values)
        {
            try
            {
                pipeline.Dispose();
            }
            catch (Exception ex)
            {
                Logs.Error("[Vision] RT-DETR pipeline dispose failed.", ex);
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
