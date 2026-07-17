using HartsyInference.Core.Backends;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Vision.Detection;

/// <summary>End-to-end RT-DETR detection pipeline. Owns an <see cref="RtDetrModel"/> + an
/// <see cref="RtDetrPreprocessor"/> (square bilinear resize + <c>/255</c>, no letterbox). RT-DETR is
/// NMS-free; post-processing is the focal-loss decode: sigmoid over the query class logits, a flat
/// top-<c>NumQueries</c> selection across (query × class), a confidence threshold, and cxcywh→xywh box
/// conversion. Boxes are returned normalized (0–1) to the source image.</summary>
public sealed class RtDetrPipeline : IDetectionPipeline
{
    private readonly IBackend _backend;
    private readonly RtDetrModel _model;
    private readonly RtDetrPreprocessor _preprocessor;
    private readonly SafeTensorsLoader? _loader;
    private readonly IReadOnlyList<string>? _labels;
    private int _disposed;

    /// <summary>Variant name (e.g. <c>"rtdetr_r18vd"</c>).</summary>
    public string ModelName => _model.Config.Name;

    /// <summary>The underlying model — exposed for diagnostics and weight preloading.</summary>
    public RtDetrModel Model => _model;

    /// <summary>Square input size fed to the backbone (640 default).</summary>
    public int InputSize => _preprocessor.TargetSize;

    /// <summary>Constructs a pipeline that loads + converts weights from a safetensors checkpoint.</summary>
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
        Dictionary<string, Tensor> converted = RtDetrCheckpointConverter.Convert(_loader.GetAllTensors());
        _model.LoadWeights(converted);
        _preprocessor = new RtDetrPreprocessor(inputSize);
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
        _preprocessor = new RtDetrPreprocessor(inputSize);
        _labels = labels;
    }

    /// <summary>Runs detection on one image. <paramref name="rgbPixels"/> is HWC-packed RGB u8.
    /// Returns detections in normalized (0-1) source-image xywh, sorted by confidence descending.</summary>
    public IReadOnlyList<DetectionResult> Detect(ReadOnlySpan<byte> rgbPixels, int width, int height, float confidenceThreshold = 0.5f)
    {
        ThrowIfDisposed();
        Tensor input = _preprocessor.Preprocess(rgbPixels, width, height);
        Tensor? cls = null;
        Tensor? boxes = null;
        try
        {
            cls = _model.Forward(_backend, input, out boxes);
            return Decode(cls, boxes, confidenceThreshold);
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

    /// <summary>Focal-loss decode: sigmoid, flat top-K over (query × class), threshold, and cxcywh→xywh
    /// (all normalized to the source image).</summary>
    private List<DetectionResult> Decode(Tensor classLogits, Tensor boxes, float confidenceThreshold)
    {
        int nq = _model.NumQueries;
        int nc = _model.NumClasses;
        ReadOnlySpan<float> cls = classLogits.AsSpan<float>();
        ReadOnlySpan<float> box = boxes.AsSpan<float>();

        int total = nq * nc;
        int[] order = new int[total];
        float[] score = new float[total];
        for (int i = 0; i < total; i++)
        {
            order[i] = i;
            score[i] = 1f / (1f + MathF.Exp(-cls[i]));
        }
        Array.Sort(order, (a, b) =>
        {
            int cmp = score[b].CompareTo(score[a]);
            return cmp != 0 ? cmp : a.CompareTo(b);
        });

        List<DetectionResult> results = new();
        int take = Math.Min(nq, total);
        for (int r = 0; r < take; r++)
        {
            int flat = order[r];
            float s = score[flat];
            if (s <= confidenceThreshold)
                break;
            int q = flat / nc;
            int c = flat % nc;
            float cx = box[q * 4 + 0];
            float cy = box[q * 4 + 1];
            float bw = box[q * 4 + 2];
            float bh = box[q * 4 + 3];
            float x1 = cx - bw * 0.5f;
            float y1 = cy - bh * 0.5f;
            results.Add(new DetectionResult
            {
                X = x1,
                Y = y1,
                Width = bw,
                Height = bh,
                Confidence = s,
                ClassIndex = c,
                Label = GetLabel(c),
            });
        }
        return results;
    }

    /// <summary>Resolves a class label — uses the supplied label table if any, else the COCO 80 fallback.</summary>
    public string GetLabel(int classId) =>
        _labels is not null && classId >= 0 && classId < _labels.Count
            ? _labels[classId]
            : CocoLabels.Get(classId);

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
