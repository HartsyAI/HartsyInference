using HartsyInference.Core.Backends;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Vision.Detection;
using HartsyInference.Vision.Face;

namespace HartsyInference.Vision.FaceDetection;

/// <summary>End-to-end YOLOv8-Face pipeline: letterboxes HWC-packed RGB u8 pixels, runs <see cref="YoloV8FaceModel"/>,
/// and returns <see cref="DetectedFace"/>s (face box + 5 landmarks) in source-image coordinates. Mirrors
/// <see cref="Detection.YoloPosePipeline"/> but yields the five ArcFace-aligned landmarks alongside each box.
///
/// <para>YOLOv8-Face is a plain YOLOv8 detector with a 1-class head plus a landmark (<c>cv4</c>) regression branch —
/// only that branch is new, everything else is the shared detection stack. <see cref="EmbedFace"/> closes the loop to
/// identity: detector → 5 landmarks → <see cref="FaceAlignment"/> → <see cref="ArcFaceModel"/> → 512-d embedding.</para></summary>
public sealed class FaceDetector : IVisionPipeline
{
    private readonly IBackend _backend;
    private readonly SafeTensorsLoader? _loader;
    private readonly YoloV8FaceModel _model;
    private readonly YoloPreprocessor _preprocessor;
    private int _disposed;

    /// <inheritdoc/>
    public string ModelName => _model.Config.Name;

    /// <summary>Underlying model — exposed for weight preloading/diagnostics.</summary>
    public YoloV8FaceModel Model => _model;

    /// <summary>Letterbox target size used by the preprocessor (640 default).</summary>
    public int InputSize => _preprocessor.TargetSize;

    /// <summary>Loads a YOLOv8-Face checkpoint from safetensors. Use a <see cref="YoloV8FaceConfig"/> preset; pass
    /// <c>kptDims: 2</c> in the preset for landmark-only checkpoints whose <c>cv4</c> branch has no visibility channel.</summary>
    public FaceDetector(IBackend backend, YoloConfig config, string safetensorsPath, int inputSize = 640)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(safetensorsPath);
        if (config.NumKeypoints != YoloV8FaceConfig.LandmarkCount)
            throw new ArgumentException($"YOLOv8-Face expects a {YoloV8FaceConfig.LandmarkCount}-point config; got NumKeypoints={config.NumKeypoints}.", nameof(config));
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException($"YOLOv8-Face checkpoint not found: {safetensorsPath}", safetensorsPath);

        _backend = backend;
        _loader = new SafeTensorsLoader();
        _loader.Load(safetensorsPath);
        _model = new YoloV8FaceModel(config);
        _model.LoadWeights(_loader.GetAllTensors());
        _preprocessor = new YoloPreprocessor(inputSize, stride: 32, stridePadAlign: false);
    }

    /// <summary>Wraps an already-weight-loaded model (e.g. for tests that inject synthetic weights). The caller owns
    /// the model's weight lifetime; this pipeline holds no safetensors handle.</summary>
    public FaceDetector(IBackend backend, YoloV8FaceModel model, int inputSize = 640)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(model);
        _backend = backend;
        _loader = null;
        _model = model;
        _preprocessor = new YoloPreprocessor(inputSize, stride: 32, stridePadAlign: false);
    }

    /// <summary>Detects faces in a single image (<paramref name="rgbPixels"/> HWC-packed RGB u8), returning each face's
    /// box and five landmarks in source-image pixels.</summary>
    public IReadOnlyList<DetectedFace> DetectFaces(
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
            (Tensor detections, Tensor landmarks) = _model.Forward(_backend, input);
            try
            {
                return FaceDetectionPostProcessor.Decode(
                    detections, landmarks, transform,
                    _model.NumClasses, _model.NumKeypoints, _model.KptDims,
                    confidenceThreshold, iouThreshold, maxDetections);
            }
            finally { detections.Dispose(); landmarks.Dispose(); }
        }
        finally { input.Dispose(); }
    }

    /// <summary>Model weights — for <c>backend.PreloadWeights</c>/<c>FreeWeights</c> around a batch of frames.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => _model.EnumerateWeights();

    /// <summary>Aligns <paramref name="face"/> to the ArcFace template and returns a <paramref name="outputSize"/>²
    /// HWC RGB u8 crop. The five landmarks are re-ordered to the template (eyes and mouth corners sorted by image-x,
    /// nose fixed) so the similarity fit is robust to a checkpoint's left/right point convention.</summary>
    public static byte[] AlignedCrop(ReadOnlySpan<byte> rgbPixels, int width, int height, DetectedFace face, int outputSize = ArcFaceModel.InputSize)
    {
        float[] pts = TemplateOrderedPoints(face.Landmarks);
        return FaceAlignment.AlignToTemplate(rgbPixels, width, height, pts, outputSize);
    }

    /// <summary>End-to-end identity embedding for one detected face: aligns the 5 landmarks to the ArcFace template,
    /// runs <paramref name="arcFace"/>, and returns the L2-normalized 512-d embedding. The detector localizes and the
    /// recognizer embeds — the reuse the pipeline is built for.</summary>
    public static float[] EmbedFace(IBackend backend, ArcFaceModel arcFace, ReadOnlySpan<byte> rgbPixels, int width, int height, DetectedFace face)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(arcFace);
        byte[] aligned = AlignedCrop(rgbPixels, width, height, face, ArcFaceModel.InputSize);
        using Tensor input = ArcFaceModel.PreprocessAligned(aligned);
        using Tensor embedding = arcFace.EmbedNormalized(backend, input);
        float[] result = new float[ArcFaceModel.EmbeddingDim];
        embedding.AsReadOnlySpan<float>().Slice(0, ArcFaceModel.EmbeddingDim).CopyTo(result);
        return result;
    }

    /// <summary>Re-orders the five landmarks into the ArcFace template rows (image-left eye, image-right eye, nose,
    /// mouth-left, mouth-right) by sorting the eye pair and mouth-corner pair on x. A checkpoint that labels its
    /// points differently still aligns correctly because the template is symmetric under this ordering.</summary>
    private static float[] TemplateOrderedPoints(Landmark5 lm)
    {
        FaceLandmark eyeLeft = lm.LeftEye.X <= lm.RightEye.X ? lm.LeftEye : lm.RightEye;
        FaceLandmark eyeRight = lm.LeftEye.X <= lm.RightEye.X ? lm.RightEye : lm.LeftEye;
        FaceLandmark mouthLeft = lm.MouthLeft.X <= lm.MouthRight.X ? lm.MouthLeft : lm.MouthRight;
        FaceLandmark mouthRight = lm.MouthLeft.X <= lm.MouthRight.X ? lm.MouthRight : lm.MouthLeft;
        return
        [
            eyeLeft.X, eyeLeft.Y,
            eyeRight.X, eyeRight.Y,
            lm.Nose.X, lm.Nose.Y,
            mouthLeft.X, mouthLeft.Y,
            mouthRight.X, mouthRight.Y,
        ];
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _loader?.Dispose();
    }
}
