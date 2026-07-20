using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Vision.Detection;

/// <summary>End-to-end YOLO11-pose pipeline — owns the safetensors mmap + <see cref="YoloV11PoseModel"/>, letterboxes
/// HWC-packed RGB u8 pixels, runs the model, and returns <see cref="PoseDetection"/>s (person box + 17 keypoints) in
/// source-image coordinates. Mirrors <see cref="YoloPipeline"/> but yields keypoints alongside the boxes.</summary>
public sealed class YoloPosePipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly SafeTensorsLoader _loader;
    private readonly YoloV11PoseModel _model;
    private readonly YoloPreprocessor _preprocessor;
    private readonly YoloConfig _config;
    private int _disposed;

    public string ModelName => _config.Name;
    public int InputSize => _preprocessor.TargetSize;

    /// <summary>Underlying model — exposed for weight preloading/diagnostics.</summary>
    public YoloV11PoseModel Model => _model;

    public YoloPosePipeline(IBackend backend, YoloConfig config, string safetensorsPath, int inputSize = 640)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(safetensorsPath);
        if (config.NumKeypoints <= 0)
            throw new ArgumentException("YoloPosePipeline requires a pose config (NumKeypoints > 0).", nameof(config));
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException($"YOLO-pose checkpoint not found: {safetensorsPath}", safetensorsPath);

        _backend = backend;
        _config = config;
        _loader = new SafeTensorsLoader();
        _loader.Load(safetensorsPath);
        _model = new YoloV11PoseModel(config);
        _model.LoadWeights(_loader.GetAllTensors());
        _preprocessor = new YoloPreprocessor(inputSize, stride: 32, stridePadAlign: false);
    }

    /// <summary>Runs pose estimation on a single image (<paramref name="rgbPixels"/> HWC-packed RGB u8).</summary>
    public IReadOnlyList<PoseDetection> Detect(
        ReadOnlySpan<byte> rgbPixels,
        int width,
        int height,
        float confidenceThreshold = 0.25f,
        float iouThreshold = 0.45f,
        int maxDetections = 300)
    {
        ThrowIfDisposed();
        (Tensor input, YoloPreprocessor.Transform transform) = _preprocessor.Preprocess(rgbPixels, width, height);
        try
        {
            (Tensor detections, Tensor keypoints) = _model.Forward(_backend, input);
            try
            {
                return YoloPosePostProcessor.Decode(
                    detections, keypoints, transform,
                    _config.NumClasses, _config.NumKeypoints, _config.KptDims,
                    confidenceThreshold, iouThreshold, maxDetections);
            }
            finally { detections.Dispose(); keypoints.Dispose(); }
        }
        finally { input.Dispose(); }
    }

    /// <summary>Model weights — for <c>backend.PreloadWeights</c>/<c>FreeWeights</c> around a batch of frames.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => _model.EnumerateWeights();

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _loader.Dispose();
    }
}
