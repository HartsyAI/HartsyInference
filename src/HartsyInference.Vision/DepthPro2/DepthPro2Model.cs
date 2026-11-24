using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.DepthPro2;

/// <summary>
/// Depth Pro 2 metric depth (upstream <c>DepthPro2.infer</c>) for one image or several views of one scene taken with one camera: the images are
/// normalised at their native size (<see cref="DepthPro2Preprocessor"/>) and run through <see cref="DepthPro2Network"/> as one clip, so the views
/// attend to each other in the cross-frame blocks (an unordered set: the result does not depend on their order), and the depth in metres is
/// <c>f_px / (W · inverseDepth)</c>. The focal length is the caller's when given, else the lower median of
/// <see cref="DepthPro2FocalEstimator"/>'s angles over the images, so every view shares one focal length as upstream's clips do. Video runs
/// through <see cref="DepthPro2VideoStream"/>.
/// </summary>
public sealed class DepthPro2Model
{
    /// <summary>The most patch tokens (16 × 16 pixels each, after padding to 32) one image may have: 2048 × 2048. The network runs at the image's own size
    /// with global attention over every token, so a larger image is refused rather than left to exhaust memory deep inside attention.</summary>
    public const int MaxPatchTokens = 16384;

    /// <summary>The most frames or views run as one joint set: upstream's training clip length. Cross-frame attention cost grows with the
    /// square of the joint token count, so longer sets are refused rather than run.</summary>
    public const int MaxJointFrames = 16;

    private readonly DepthPro2Network _network;

    public DepthPro2Model(DepthPro2Network network)
    {
        ArgumentNullException.ThrowIfNull(network);
        _network = network;
    }

    public DepthPro2Network Network => _network;

    /// <summary>The maps of <paramref name="images"/> (RGB24, all <paramref name="width"/> × <paramref name="height"/>), in order.</summary>
    /// <param name="focalLengthPx">The focal length in pixels, or null to estimate it with <paramref name="focalEstimator"/>.</param>
    /// <param name="focalEstimator">Required when <paramref name="focalLengthPx"/> is null; unused otherwise.</param>
    /// <param name="keepInverseDepth">False converts each inverse-depth buffer to metres in place and leaves <see cref="DepthPro2Map.InverseDepth"/> null,
    /// saving one H×W buffer per image.</param>
    /// <exception cref="ArgumentException">The image is over <see cref="MaxPatchTokens"/>.</exception>
    public DepthPro2Prediction Predict(IBackend backend, IReadOnlyList<byte[]> images, int width, int height, double? focalLengthPx,
        DepthPro2FocalEstimator? focalEstimator, CancellationToken cancel = default, bool keepInverseDepth = true)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(images);
        if (images.Count == 0) throw new ArgumentException("Name at least one image.", nameof(images));
        if (images.Count > MaxJointFrames)
            throw new ArgumentException($"At most {MaxJointFrames} views run together; got {images.Count}.", nameof(images));
        CheckSize(width, height);
        CheckFocal(focalLengthPx, focalEstimator);

        cancel.ThrowIfCancellationRequested();
        float[][] inverse;
        using (Tensor frames = DepthPro2Preprocessor.NormalizeFrames(images, width, height))
        using (Tensor maps = _network.PredictInverseDepthFrames(backend, frames))
            inverse = SplitFrames(maps, images.Count);

        double focal = focalLengthPx ?? EstimateFocal(backend, images, images.Count, width, height, focalEstimator!, cancel);
        DepthPro2Map[] result = new DepthPro2Map[images.Count];
        for (int i = 0; i < images.Count; i++) result[i] = ToMap(inverse[i], width, focal, keepInverseDepth);
        return new DepthPro2Prediction(result, focal, focalLengthPx is null, width, height);
    }

    /// <summary>Throws when a <paramref name="width"/> × <paramref name="height"/> image is over <see cref="MaxPatchTokens"/> once padded to 32.</summary>
    public static void CheckSize(int width, int height)
    {
        long padH = (height + DepthPro2Preset.InputMultiple - 1L) / DepthPro2Preset.InputMultiple * DepthPro2Preset.InputMultiple;
        long padW = (width + DepthPro2Preset.InputMultiple - 1L) / DepthPro2Preset.InputMultiple * DepthPro2Preset.InputMultiple;
        long tokens = padH / 16 * (padW / 16);
        if (tokens > MaxPatchTokens)
        {
            throw new ArgumentException(
                $"A {width}x{height} image is {tokens} patch tokens; Depth Pro 2 runs at the image's own size with global attention and this build takes at "
                + $"most {MaxPatchTokens} (2048x2048 or an equal area). Downscale the image first.");
        }
    }

    /// <summary>
    /// The focal length a video gets when none is given (upstream <c>infer_video</c>): the lower median of the estimator's angles over the first
    /// <paramref name="window"/> frames (all of them when fewer), kept for every frame.
    /// </summary>
    public static double EstimateFocal(IBackend backend, IReadOnlyList<byte[]> frames, int window, int width, int height,
        DepthPro2FocalEstimator focalEstimator, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(frames);
        ArgumentNullException.ThrowIfNull(focalEstimator);
        int count = Math.Min(window, frames.Count);
        if (count < 1) throw new ArgumentException("Name at least one frame.", nameof(frames));
        DepthPro2FieldOfView[] angles = new DepthPro2FieldOfView[count];
        for (int i = 0; i < count; i++)
        {
            cancel.ThrowIfCancellationRequested();
            using Tensor pixels = DepthPro2Preprocessor.Normalize(frames[i], width, height);
            angles[i] = focalEstimator.Estimate(backend, pixels);
        }

        return DepthPro2FocalEstimator.FocalLengthPx(angles, height, width);
    }

    /// <summary>The metric map of one frame's inverse depth (taken over) at <paramref name="focalLengthPx"/>; without
    /// <paramref name="keepInverseDepth"/> the buffer is converted to metres in place.</summary>
    public static DepthPro2Map ToMap(float[] inverseDepth, int width, double focalLengthPx, bool keepInverseDepth = true)
    {
        ArgumentNullException.ThrowIfNull(inverseDepth);
        float[] depth = keepInverseDepth ? new float[inverseDepth.Length] : inverseDepth, valid = new float[inverseDepth.Length];
        DepthPro2Preprocessor.ToMetric(inverseDepth, width, focalLengthPx, depth, valid);
        return new DepthPro2Map(keepInverseDepth ? inverseDepth : null, depth, valid);
    }

    /// <summary>Each frame of <c>[T, 1, H, W]</c> as its own array.</summary>
    internal static float[][] SplitFrames(Tensor maps, int count)
    {
        int plane = (int)(maps.ElementCount / count);
        ReadOnlySpan<float> all = maps.AsReadOnlySpan<float>();
        float[][] frames = new float[count][];
        for (int i = 0; i < count; i++) frames[i] = all.Slice(i * plane, plane).ToArray();
        return frames;
    }

    private static void CheckFocal(double? focalLengthPx, DepthPro2FocalEstimator? focalEstimator)
    {
        if (focalLengthPx is { } given && (!double.IsFinite(given) || given <= 0))
            throw new ArgumentOutOfRangeException(nameof(focalLengthPx), $"The focal length must be a positive number of pixels; got {given}.");
        if (focalLengthPx is null && focalEstimator is null)
            throw new ArgumentException("Give a focal length or a focal estimator.", nameof(focalEstimator));
    }
}
