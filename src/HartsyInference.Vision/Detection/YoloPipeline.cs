using HartsyInference.Core.Backends;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Vision.Detection;

/// <summary>End-to-end YOLO object detection pipeline. Owns the safetensors mmap + <see cref="YoloModel"/>
/// instance; constructs a <see cref="YoloPreprocessor"/> with the requested input size. Callers
/// hand in HWC-packed RGB u8 pixels and receive a list of <see cref="YoloDetection"/> in
/// source-image coordinates.
/// <para>Implements <see cref="IDetectionPipeline"/> for DI registration — the byte-stream
/// overload routes through <see cref="PngDecoder"/>, so PNG inputs work out of the box. JPEG /
/// WebP inputs need an external decoder (caller decodes, then passes raw RGB pixels via
/// <see cref="Detect(ReadOnlySpan{byte}, int, int, float, float, int, bool)"/>).</para></summary>
public sealed class YoloPipeline : IDetectionPipeline
{
    private readonly IBackend _backend;
    private readonly SafeTensorsLoader _loader;
    private readonly IYoloDetectModel _model;
    private readonly YoloPreprocessor _preprocessor;
    private readonly IReadOnlyList<string>? _labels;
    private int _disposed;

    /// <summary>Variant name (e.g. <c>"yolov8n"</c>, <c>"yolo11n"</c>).</summary>
    public string ModelName => _model.Config.Name;

    /// <summary>The underlying model — exposed for diagnostics and weight preloading.</summary>
    public IYoloDetectModel Model => _model;

    /// <summary>Letterbox target size used by the preprocessor (640 default).</summary>
    public int InputSize => _preprocessor.TargetSize;

    /// <summary>Constructs a YOLOv8 pipeline. Use <see cref="LoadV11"/> for YOLO11 checkpoints — the
    /// architecture lives in the model class, not the config, so we need a separate factory.</summary>
    public YoloPipeline(IBackend backend, YoloConfig config, string safetensorsPath, int inputSize = 640, IReadOnlyList<string>? labels = null)
        : this(backend, new YoloModel(config), config, safetensorsPath, inputSize, labels)
    {
    }

    /// <summary>Factory for YOLO11 checkpoints. Use the <c>YoloV11n</c>/<c>YoloV11s</c>/etc.
    /// presets on <see cref="YoloConfig"/>.</summary>
    public static YoloPipeline LoadV11(IBackend backend, YoloConfig config, string safetensorsPath, int inputSize = 640, IReadOnlyList<string>? labels = null)
    {
        return new YoloPipeline(backend, new YoloV11Model(config), config, safetensorsPath, inputSize, labels);
    }

    private YoloPipeline(IBackend backend, IYoloDetectModel model, YoloConfig config, string safetensorsPath, int inputSize, IReadOnlyList<string>? labels)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(safetensorsPath);
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException($"YOLO checkpoint not found: {safetensorsPath}", safetensorsPath);

        _backend = backend;
        _loader = new SafeTensorsLoader();
        _loader.Load(safetensorsPath);

        _model = model;
        Dictionary<string, Tensor> weights = _loader.GetAllTensors();
        _model.LoadWeights(weights);

        _preprocessor = new YoloPreprocessor(inputSize, stride: 32, stridePadAlign: false);
        _labels = labels;
    }

    /// <summary>Runs detection on a single image. <paramref name="rgbPixels"/> is HWC-packed RGB u8.</summary>
    public IReadOnlyList<YoloDetection> Detect(
        ReadOnlySpan<byte> rgbPixels,
        int width,
        int height,
        float confidenceThreshold = 0.25f,
        float iouThreshold = 0.45f,
        int maxDetections = 300,
        bool classAgnostic = false)
    {
        ThrowIfDisposed();
        (Tensor input, YoloPreprocessor.Transform transform) = _preprocessor.Preprocess(rgbPixels, width, height);
        try
        {
            using Tensor output = _model.Forward(_backend, input);
            return YoloPostProcessor.Decode(
                output, transform, _model.NumClasses,
                confidenceThreshold, iouThreshold, maxDetections, classAgnostic);
        }
        finally { input.Dispose(); }
    }

    /// <summary>Returns the label string for a detection's class id — uses the supplied label
    /// table if any, falling back to <see cref="CocoLabels.Get"/>.</summary>
    public string GetLabel(int classId) =>
        _labels is not null && classId >= 0 && classId < _labels.Count ? _labels[classId] : CocoLabels.Get(classId);

    /// <summary><see cref="IDetectionPipeline"/> implementation — decodes PNG bytes, runs YOLO,
    /// converts <see cref="YoloDetection"/> (xyxy pixel coords) into <see cref="DetectionResult"/>
    /// (normalized xywh + label string). Only PNG is supported by the built-in decoder; for JPEG
    /// or WebP, decode externally and call <see cref="Detect(ReadOnlySpan{byte}, int, int, float, float, int, bool)"/> directly.</summary>
    public Task<IReadOnlyList<DetectionResult>> DetectAsync(ReadOnlyMemory<byte> imageBytes, float confidenceThreshold = 0.5f, CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();

        (byte[] rgb, int width, int height) = PngDecoder.Decode(imageBytes.Span);
        IReadOnlyList<YoloDetection> raw = Detect(rgb, width, height, confidenceThreshold);

        DetectionResult[] results = new DetectionResult[raw.Count];
        float invW = 1f / width;
        float invH = 1f / height;
        for (int i = 0; i < raw.Count; i++)
        {
            YoloDetection d = raw[i];
            results[i] = new DetectionResult
            {
                X = d.X1 * invW,
                Y = d.Y1 * invH,
                Width = (d.X2 - d.X1) * invW,
                Height = (d.Y2 - d.Y1) * invH,
                Confidence = d.Confidence,
                ClassIndex = d.ClassId,
                Label = GetLabel(d.ClassId),
            };
        }
        return Task.FromResult<IReadOnlyList<DetectionResult>>(results);
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    /// <summary>Releases the safetensors mmap and clears references to the model.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _loader.Dispose();
    }
}
