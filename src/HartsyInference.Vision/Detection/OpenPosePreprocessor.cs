using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Codec;

namespace HartsyInference.Vision.Detection;

/// <summary>ControlNet OpenPose conditioning preprocessor — runs the YOLO11-pose detector on a decoded RGB image
/// and renders every detected person through <see cref="OpenPoseRenderer"/> into the <c>[1,3,H,W]</c> F32 [0,1]
/// skeleton image (colored limbs/joints on black) that control_v11p_sd15_openpose / SDXL-openpose ControlNets
/// expect. Host code by design: runs once per generation, outside the per-step hot path.</summary>
public sealed class OpenPosePreprocessor : IDisposable
{
    private readonly YoloPosePipeline _pipeline;
    private readonly bool _ownsPipeline;
    private int _disposed;

    /// <summary>Loads a yolo11n-pose checkpoint and owns the resulting pipeline.</summary>
    public OpenPosePreprocessor(IBackend backend, string safetensorsPath, int inputSize = 640)
    {
        _pipeline = new YoloPosePipeline(backend, YoloConfig.YoloV11nPose, safetensorsPath, inputSize);
        _ownsPipeline = true;
    }

    /// <summary>Wraps an already-loaded pose pipeline (shared with e.g. the Wan-Animate skeleton flow).
    /// The caller retains ownership — <see cref="Dispose"/> will not dispose it.</summary>
    public OpenPosePreprocessor(YoloPosePipeline pipeline)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        _pipeline = pipeline;
        _ownsPipeline = false;
    }

    /// <summary>Underlying pose pipeline — exposed for weight preloading and direct keypoint access.</summary>
    public YoloPosePipeline Pipeline => _pipeline;

    /// <summary>Detects all people in <paramref name="rgbPixels"/> (HWC-packed RGB u8) and returns the OpenPose
    /// skeleton render as <c>[1, 3, outputHeight, outputWidth]</c> F32 in [0,1]. Output dimensions default to the
    /// source dimensions; keypoints are rescaled when they differ (e.g. conditioning at gen resolution). No people
    /// detected yields an all-black image, which conditions ControlNet to "no pose constraint".</summary>
    public Tensor Process(
        ReadOnlySpan<byte> rgbPixels,
        int width,
        int height,
        int outputWidth = 0,
        int outputHeight = 0,
        float confidenceThreshold = 0.25f,
        float visThreshold = 0.3f)
    {
        ThrowIfDisposed();
        IReadOnlyList<PoseDetection> people = _pipeline.Detect(rgbPixels, width, height, confidenceThreshold);
        return RenderToTensor(
            people, width, height,
            outputWidth > 0 ? outputWidth : width,
            outputHeight > 0 ? outputHeight : height,
            visThreshold);
    }

    /// <summary>Renders already-detected people (keypoints in source-image pixels of a
    /// <paramref name="sourceWidth"/>×<paramref name="sourceHeight"/> image) to the ControlNet conditioning tensor
    /// <c>[1, 3, outputHeight, outputWidth]</c> F32 in [0,1]. Pure raster — unit-testable without weights.</summary>
    public static Tensor RenderToTensor(
        IReadOnlyList<PoseDetection> people,
        int sourceWidth,
        int sourceHeight,
        int outputWidth,
        int outputHeight,
        float visThreshold = 0.3f)
    {
        if (sourceWidth <= 0 || sourceHeight <= 0)
            throw new ArgumentException($"Source dimensions must be positive; got {sourceWidth}x{sourceHeight}.");
        byte[] canvas = OpenPoseRenderer.RenderBodyPose(
            people, outputWidth, outputHeight,
            (float)outputWidth / sourceWidth, (float)outputHeight / sourceHeight, visThreshold);
        return ImageTensor.RgbToTensor01(canvas, outputWidth, outputHeight);
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        if (_ownsPipeline)
            _pipeline.Dispose();
    }
}
