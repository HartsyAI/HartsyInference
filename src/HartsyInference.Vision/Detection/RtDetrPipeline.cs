using HartsyInference.Core.Backends;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Vision.Detection;

/// <summary>End-to-end RT-DETR detection pipeline. Owns an <see cref="RtDetrModel"/> + a
/// <see cref="YoloPreprocessor"/> (reused: RT-DETR uses the same letterbox + /255 normalize as YOLO).
/// RT-DETR is NMS-free, so post-processing is just sigmoid over the query class logits + a confidence
/// threshold + score-descending sort of the (already capped at <c>NumQueries</c>) candidates.
/// <para>Implements <see cref="IDetectionPipeline"/>; the byte-stream overload routes through
/// <see cref="PngDecoder"/> so PNG inputs work out of the box. JPEG/WebP callers decode externally
/// and call <see cref="Detect(ReadOnlySpan{byte}, int, int, float)"/> with raw RGB.</para></summary>
public sealed class RtDetrPipeline : IDetectionPipeline
{
    private readonly IBackend _backend;
    private readonly RtDetrModel _model;
    private readonly YoloPreprocessor _preprocessor;
    private readonly SafeTensorsLoader? _loader;
    private readonly IReadOnlyList<string>? _labels;
    private int _disposed;

    /// <summary>Variant name (e.g. <c>"rtdetr-r50"</c>).</summary>
    public string ModelName => _model.Config.Name;

    /// <summary>The underlying model — exposed for diagnostics and weight preloading.</summary>
    public RtDetrModel Model => _model;

    /// <summary>Letterbox target size fed to the backbone (640 default).</summary>
    public int InputSize => _preprocessor.TargetSize;

    /// <summary>Constructs a pipeline that loads weights from a safetensors checkpoint.</summary>
    public RtDetrPipeline(IBackend backend, RtDetrConfig config, string safetensorsPath, int inputSize = 640, IReadOnlyList<string>? labels = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(safetensorsPath);
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException($"RT-DETR checkpoint not found: {safetensorsPath}", safetensorsPath);

        _backend = backend;
        _model = new RtDetrModel(config);
        _loader = new SafeTensorsLoader();
        _loader.Load(safetensorsPath);
        _model.LoadWeights(_loader.GetAllTensors());
        _preprocessor = new YoloPreprocessor(inputSize, stride: 32, stridePadAlign: false);
        _labels = labels;
    }

    /// <summary>Constructs a pipeline around an already-built + weight-loaded model (custom loaders / tests).</summary>
    public RtDetrPipeline(IBackend backend, RtDetrModel model, int inputSize = 640, IReadOnlyList<string>? labels = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(model);
        _backend = backend;
        _model = model;
        _loader = null;
        _preprocessor = new YoloPreprocessor(inputSize, stride: 32, stridePadAlign: false);
        _labels = labels;
    }

    /// <summary>Runs detection on one image. <paramref name="rgbPixels"/> is HWC-packed RGB u8.
    /// Returns detections in normalized (0-1) source-image xywh, sorted by confidence descending.</summary>
    public IReadOnlyList<DetectionResult> Detect(ReadOnlySpan<byte> rgbPixels, int width, int height, float confidenceThreshold = 0.5f)
    {
        ThrowIfDisposed();
        (Tensor input, YoloPreprocessor.Transform transform) = _preprocessor.Preprocess(rgbPixels, width, height);
        Tensor? cls = null;
        Tensor? boxes = null;
        try
        {
            (cls, boxes) = _model.Forward(_backend, input);
            return Decode(cls, boxes, transform, confidenceThreshold);
        }
        finally
        {
            input.Dispose();
            cls?.Dispose();
            boxes?.Dispose();
        }
    }

    /// <summary><see cref="IDetectionPipeline"/> implementation — decodes PNG bytes, runs RT-DETR,
    /// returns normalized xywh <see cref="DetectionResult"/>. PNG only via the built-in decoder.</summary>
    public Task<IReadOnlyList<DetectionResult>> DetectAsync(ReadOnlyMemory<byte> imageBytes, float confidenceThreshold = 0.5f, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        (byte[] rgb, int width, int height) = PngDecoder.Decode(imageBytes.Span);
        IReadOnlyList<DetectionResult> results = Detect(rgb, width, height, confidenceThreshold);
        return Task.FromResult(results);
    }

    /// <summary>Sigmoid + threshold + convert boxes from letterbox-normalized cxcywh to source-image normalized xywh.</summary>
    private List<DetectionResult> Decode(Tensor classLogits, Tensor boxes, YoloPreprocessor.Transform transform, float confidenceThreshold)
    {
        int nq = _model.NumQueries;
        int nc = _model.NumClasses;
        ReadOnlySpan<float> cls = classLogits.AsSpan<float>();
        ReadOnlySpan<float> box = boxes.AsSpan<float>();

        float canvasW = transform.PaddedWidth;
        float canvasH = transform.PaddedHeight;
        float invSrcW = 1f / transform.SourceWidth;
        float invSrcH = 1f / transform.SourceHeight;

        List<DetectionResult> results = new();
        for (int q = 0; q < nq; q++)
        {
            int bestClass = 0;
            float bestScore = 0f;
            int baseIdx = q * nc;
            for (int c = 0; c < nc; c++)
            {
                float score = Sigmoid(cls[baseIdx + c]);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = c;
                }
            }
            if (bestScore < confidenceThreshold)
                continue;

            float cx = box[q * 4 + 0] * canvasW;
            float cy = box[q * 4 + 1] * canvasH;
            float bw = box[q * 4 + 2] * canvasW;
            float bh = box[q * 4 + 3] * canvasH;
            (float sx1, float sy1, float sx2, float sy2) = transform.Invert(cx - bw * 0.5f, cy - bh * 0.5f, cx + bw * 0.5f, cy + bh * 0.5f);

            results.Add(new DetectionResult
            {
                X = sx1 * invSrcW,
                Y = sy1 * invSrcH,
                Width = (sx2 - sx1) * invSrcW,
                Height = (sy2 - sy1) * invSrcH,
                Confidence = bestScore,
                ClassIndex = bestClass,
                Label = GetLabel(bestClass),
            });
        }
        results.Sort((a, b) => b.Confidence.CompareTo(a.Confidence));
        return results;
    }

    /// <summary>Resolves a class label — uses the supplied label table if any, else the COCO 80 fallback.</summary>
    public string GetLabel(int classId) =>
        _labels is not null && classId >= 0 && classId < _labels.Count
            ? _labels[classId]
            : CocoLabels.Get(classId);

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    /// <summary>Releases the safetensors mmap (if this pipeline owns one).</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _loader?.Dispose();
    }
}
