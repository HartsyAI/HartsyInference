using HartsyInference.Core.Backends;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Vision.Detection;

/// <summary>End-to-end YOLO instance segmentation pipeline. Produces <see cref="YoloSegResult"/>
/// per detection — a bounding box, class, confidence, AND a binary mask in source-image
/// coordinates. Owns the safetensors mmap and <see cref="YoloSegModel"/>; uses
/// <see cref="YoloPreprocessor"/> for letterbox + normalize and <see cref="MaskAssembly.AssembleMask"/>
/// for the per-detection mask combine-sigmoid-resize step.</summary>
public sealed class YoloSegPipeline : IVisionPipeline
{
    private readonly IBackend _backend;
    private readonly SafeTensorsLoader _loader;
    private readonly YoloSegModel _model;
    private readonly YoloPreprocessor _preprocessor;
    private readonly IReadOnlyList<string>? _labels;
    private int _disposed;

    /// <summary>Variant name (e.g. <c>"yolov8n-seg"</c>).</summary>
    public string ModelName { get; }

    /// <summary>Underlying segmentation model.</summary>
    public YoloSegModel Model => _model;

    /// <summary>Letterbox target size.</summary>
    public int InputSize => _preprocessor.TargetSize;

    /// <summary>Constructs a YOLOv8-seg pipeline from a folded safetensors checkpoint.</summary>
    public YoloSegPipeline(IBackend backend, YoloConfig config, string safetensorsPath, int inputSize = 640, int numMasks = 32, IReadOnlyList<string>? labels = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(safetensorsPath);
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException($"YOLO seg checkpoint not found: {safetensorsPath}", safetensorsPath);

        _backend = backend;
        _loader = new SafeTensorsLoader();
        _loader.Load(safetensorsPath);

        _model = new YoloSegModel(config, numMasks);
        Dictionary<string, Tensor> weights = _loader.GetAllTensors();
        _model.LoadWeights(weights);

        _preprocessor = new YoloPreprocessor(inputSize, stride: 32, stridePadAlign: false);
        _labels = labels;
        ModelName = $"{config.Name}-seg";
    }

    /// <summary>Returns the label string for a class id (uses supplied label table or COCO).</summary>
    public string GetLabel(int classId) =>
        _labels is not null && classId >= 0 && classId < _labels.Count
            ? _labels[classId]
            : CocoLabels.Get(classId);

    /// <summary>Runs detection + segmentation on a single image. <paramref name="rgbPixels"/> is
    /// HWC-packed RGB u8.</summary>
    public IReadOnlyList<YoloSegResult> Segment(
        ReadOnlySpan<byte> rgbPixels,
        int width,
        int height,
        float confidenceThreshold = 0.25f,
        float iouThreshold = 0.45f,
        int maxDetections = 300,
        bool classAgnostic = false,
        float maskThreshold = 0.5f)
    {
        ThrowIfDisposed();
        (Tensor input, YoloPreprocessor.Transform transform) = _preprocessor.Preprocess(rgbPixels, width, height);
        try
        {
            (Tensor detections, Tensor protos) = _model.ForwardWithProtos(_backend, input);
            try
            {
                // The detections tensor is [1, 4 + nc + nm, totalAnchors]. The post-processor for
                // segmentation needs the mask coefficients alongside each surviving detection, so
                // we walk the anchors here rather than reusing YoloPostProcessor.Decode.
                int totalAnchors = (int)detections.Shape[2];
                int channels = (int)detections.Shape[1];
                int nc = _model.NumClasses;
                int nm = _model.NumMasks;
                if (channels != 4 + nc + nm)
                    throw new InvalidOperationException($"Unexpected channel count: {channels} != 4+{nc}+{nm}.");

                ReadOnlySpan<float> data = detections.AsReadOnlySpan<float>();
                List<(YoloDetection det, float[] coefs)> candidates = new(capacity: 256);

                for (int a = 0; a < totalAnchors; a++)
                {
                    float bestScore = 0f;
                    int bestClass = -1;
                    for (int k = 0; k < nc; k++)
                    {
                        float p = data[(4 + k) * totalAnchors + a];
                        if (p > bestScore) { bestScore = p; bestClass = k; }
                    }
                    if (bestScore < confidenceThreshold || bestClass < 0) continue;

                    float cx = data[0 * totalAnchors + a];
                    float cy = data[1 * totalAnchors + a];
                    float bw = data[2 * totalAnchors + a];
                    float bh = data[3 * totalAnchors + a];
                    (float x1, float y1, float x2, float y2) = transform.Invert(
                        cx - bw * 0.5f, cy - bh * 0.5f, cx + bw * 0.5f, cy + bh * 0.5f);

                    float[] coefs = new float[nm];
                    for (int k = 0; k < nm; k++)
                        coefs[k] = data[(4 + nc + k) * totalAnchors + a];

                    candidates.Add((new YoloDetection(x1, y1, x2, y2, bestScore, bestClass), coefs));
                }

                // NMS on the candidates' detection part (mask coefs go along for the ride).
                // We sort by confidence + greedy IoU suppression — same algorithm as
                // NonMaxSuppression.Run but inlined here so we can keep the coefficient pairings.
                candidates.Sort((a, b) => b.det.Confidence.CompareTo(a.det.Confidence));
                bool[] suppressed = new bool[candidates.Count];
                List<YoloSegResult> results = new(Math.Min(maxDetections, candidates.Count));
                for (int i = 0; i < candidates.Count; i++)
                {
                    if (suppressed[i]) continue;
                    YoloDetection pickDet = candidates[i].det;
                    byte[] mask = MaskAssembly.AssembleMask(pickDet, candidates[i].coefs, protos, transform, maskThreshold);
                    results.Add(new YoloSegResult
                    {
                        Detection = pickDet,
                        Mask = mask,
                        Width = transform.SourceWidth,
                        Height = transform.SourceHeight,
                    });
                    if (results.Count >= maxDetections) break;

                    for (int j = i + 1; j < candidates.Count; j++)
                    {
                        if (suppressed[j]) continue;
                        if (!classAgnostic && candidates[j].det.ClassId != pickDet.ClassId) continue;
                        if (pickDet.Iou(candidates[j].det) > iouThreshold) suppressed[j] = true;
                    }
                }

                return results;
            }
            finally
            {
                detections.Dispose();
                protos.Dispose();
            }
        }
        finally { input.Dispose(); }
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _loader.Dispose();
    }
}
