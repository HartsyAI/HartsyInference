using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.DepthPro2;

/// <summary>
/// Depth Pro 2 metric depth (upstream <c>DepthPro2.infer</c>) for one image or several views of one scene taken with one camera: each image is normalised at
/// its native size (<see cref="DepthPro2Preprocessor"/>), <see cref="DepthPro2Network"/> predicts focal-normalised inverse depth, and the depth in metres is
/// <c>f_px / (W · inverseDepth)</c>. The focal length is the caller's when given, else the lower median of <see cref="DepthPro2FocalEstimator"/>'s angles over
/// the images, so every view shares one focal length as upstream's clips do. Each view's depth is predicted on its own: the cross-frame attention upstream
/// applies between the images of a clip is not implemented yet.
/// </summary>
public sealed class DepthPro2Model
{
    /// <summary>The most patch tokens (16 × 16 pixels each, after padding to 32) one image may have: 2048 × 2048. The network runs at the image's own size
    /// with global attention over every token, so a larger image is refused rather than left to exhaust memory deep inside attention.</summary>
    public const int MaxPatchTokens = 16384;

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
        CheckSize(width, height);
        if (focalLengthPx is { } given && (!double.IsFinite(given) || given <= 0))
            throw new ArgumentOutOfRangeException(nameof(focalLengthPx), $"The focal length must be a positive number of pixels; got {given}.");
        if (focalLengthPx is null && focalEstimator is null)
            throw new ArgumentException("Give a focal length or a focal estimator.", nameof(focalEstimator));

        float[][] inverse = new float[images.Count][];
        DepthPro2FieldOfView[]? angles = focalLengthPx is null ? new DepthPro2FieldOfView[images.Count] : null;
        for (int i = 0; i < images.Count; i++)
        {
            cancel.ThrowIfCancellationRequested();
            using Tensor pixels = DepthPro2Preprocessor.Normalize(images[i], width, height);
            using (Tensor map = _network.PredictInverseDepth(backend, pixels))
                inverse[i] = map.AsReadOnlySpan<float>().ToArray();
            if (angles is not null) angles[i] = focalEstimator!.Estimate(backend, pixels);
        }

        double focal = focalLengthPx ?? DepthPro2FocalEstimator.FocalLengthPx(angles!, height, width);
        DepthPro2Map[] maps = new DepthPro2Map[images.Count];
        for (int i = 0; i < images.Count; i++)
        {
            float[] depth = keepInverseDepth ? new float[inverse[i].Length] : inverse[i], valid = new float[inverse[i].Length];
            DepthPro2Preprocessor.ToMetric(inverse[i], width, focal, depth, valid);
            maps[i] = new DepthPro2Map(keepInverseDepth ? inverse[i] : null, depth, valid);
        }

        return new DepthPro2Prediction(maps, focal, focalLengthPx is null, width, height);
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
}
