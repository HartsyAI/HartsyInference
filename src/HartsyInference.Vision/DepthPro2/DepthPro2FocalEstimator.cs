using HartsyInference.Core.Backends;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.Dinov3;

namespace HartsyInference.Vision.DepthPro2;

/// <summary>
/// Depth Pro 2's focal estimator (upstream <c>FOVEstimator</c>, checkpoint <c>apple/DepthPro2-focal</c>): the frame is resized (antialiased bilinear) so its
/// short side is at most 512 and both sides are multiples of 32, centre-cropped when wider than 16:9, encoded by a DINOv3 ViT-S/16 trunk (blocks 2, 5, 8 and
/// 11 tapped with their prefix tokens; blocks 5, 7, 9 and 11 restore the prefix, as a lone frame does upstream), and <see cref="DepthPro2FovHead"/> regresses
/// the field of view as <c>1 + sigmoid(logit)·178</c> degrees, converted back to the full frame after a crop. Weights are borrowed from the caller, F32.
/// </summary>
public sealed class DepthPro2FocalEstimator
{
    private const int MaxShortSide = 512;
    private const int SizeMultiple = 32;
    private const double MaxAspect = 1.7778;   // upstream fov.py's MAX_ASPECT literal, not 16.0 / 9; keep it as written
    private const double MinFovDegrees = 1.0, MaxFovDegrees = 179.0;
    private static readonly int[] TappedBlocks = [2, 5, 8, 11];
    private static readonly int[] PrefixResetBlocks = [5, 7, 9, 11];

    private readonly Dinov2Preset _trunkPreset;
    private readonly Dinov3VisionEncoder _trunk;
    private readonly DepthPro2FovHead _head;

    /// <param name="weights">The tensors by upstream name (<c>trunk.*</c>, <c>head.*</c>).</param>
    /// <exception cref="InvalidOperationException">A weight is missing or has the wrong shape.</exception>
    public DepthPro2FocalEstimator(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _trunkPreset = DepthPro2Preset.TimmBackbone(Dinov2Preset.V3Small16);
        _trunk = new Dinov3VisionEncoder(_trunkPreset);
        _trunk.LoadWeights(Dinov3TimmKeys.ToHuggingFace(weights, "trunk.", _trunkPreset));
        _head = new DepthPro2FovHead(weights, "head.", _trunkPreset.HiddenSize, TappedBlocks.Length, 1 + _trunkPreset.NumRegisterTokens);
    }

    /// <summary>The trunk's and the head's tensors.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => _trunk.EnumerateWeights().Concat(_head.EnumerateWeights());

    /// <summary>
    /// The field of view of normalised pixels <c>[1, 3, H, W]</c> (unpadded). <paramref name="observe"/> sees <c>fov_input</c> (the resized, cropped frame),
    /// <c>fov_taps</c>, the head's stages and <c>fov_logits</c>.
    /// </summary>
    public DepthPro2FieldOfView Estimate(IBackend backend, Tensor pixels, Action<string, Tensor>? observe = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(pixels);
        if (pixels.DType != DType.F32 || pixels.Shape.Rank != 4 || pixels.Shape[0] != 1 || pixels.Shape[1] != 3 || pixels.Shape[2] < 1 || pixels.Shape[3] < 1)
            throw new ArgumentException($"pixels must be F32 [1, 3, H, W]; got {pixels.DType} {pixels.Shape}.", nameof(pixels));
        int height = (int)pixels.Shape[2], width = (int)pixels.Shape[3];
        (int fullH, int fullW) = InputSize(height, width);
        (int cropH, int cropW, int offH, int offW) = CenterCrop(fullH, fullW);

        using Tensor input = ResizeAndCrop(pixels, fullH, fullW, cropH, cropW, offH, offW);
        observe?.Invoke("fov_input", input);
        int patch = _trunkPreset.PatchSize, gridH = cropH / patch, gridW = cropW / patch;
        using Tensor taps = EncodeTaps(backend, input);
        observe?.Invoke("fov_taps", taps);
        using Tensor logits = _head.Forward(backend, taps, gridH, gridW, observe);
        observe?.Invoke("fov_logits", logits);

        ReadOnlySpan<float> l = logits.AsReadOnlySpan<float>();
        float hfov = (float)MinFovDegrees + Sigmoid(l[0]) * (float)(MaxFovDegrees - MinFovDegrees);
        float vfov = (float)MinFovDegrees + Sigmoid(l[1]) * (float)(MaxFovDegrees - MinFovDegrees);
        if (cropH == fullH && cropW == fullW) return new DepthPro2FieldOfView(hfov, vfov);
        return CropAnglesToFull(new DepthPro2FieldOfView(hfov, vfov), cropH * (double)height / fullH, cropW * (double)width / fullW, height, width);
    }

    /// <summary>The focal length in pixels of one frame of normalised pixels <c>[1, 3, H, W]</c>.</summary>
    public double FocalLengthPx(IBackend backend, Tensor pixels) =>
        FocalLengthPx([Estimate(backend, pixels)], (int)pixels.Shape[2], (int)pixels.Shape[3]);

    /// <summary>Upstream <c>median_focal_length_px</c>: the lower median of each angle over the frames, then the focal length from the angle of the shorter
    /// side (horizontal when the frame is portrait).</summary>
    public static double FocalLengthPx(IReadOnlyList<DepthPro2FieldOfView> frames, int height, int width)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frames.Count == 0) throw new ArgumentException("Name at least one frame.", nameof(frames));
        double[] h = new double[frames.Count], v = new double[frames.Count];
        for (int i = 0; i < frames.Count; i++)
        {
            h[i] = frames[i].HorizontalDegrees;
            v[i] = frames[i].VerticalDegrees;
        }

        Array.Sort(h);
        Array.Sort(v);
        int lower = (frames.Count - 1) / 2;
        return width < height
            ? 0.5 * width / Math.Tan(h[lower] * Math.PI / 180.0 / 2.0)
            : 0.5 * height / Math.Tan(v[lower] * Math.PI / 180.0 / 2.0);
    }

    /// <summary>The resize target of a frame before any crop: the short side at most 512, both sides rounded to multiples of 32. A frame is
    /// not upsampled, except that each side is at least 32.</summary>
    public static (int Height, int Width) InputSize(int height, int width)
    {
        if (height <= 0 || width <= 0) throw new ArgumentOutOfRangeException(nameof(height), "The frame size must be positive.");
        int shortSide = Math.Min(height, width);
        int target = Math.Max(SizeMultiple, Math.Min(MaxShortSide, shortSide / SizeMultiple * SizeMultiple));
        double scale = (double)target / shortSide;
        // Python's round: half to even.
        return (Math.Max(SizeMultiple, (int)Math.Round(height * scale / SizeMultiple, MidpointRounding.ToEven) * SizeMultiple),
            Math.Max(SizeMultiple, (int)Math.Round(width * scale / SizeMultiple, MidpointRounding.ToEven) * SizeMultiple));
    }

    /// <summary>The centre crop of a resized frame: the long side cut to at most 16:9 of the short one, in multiples of 32.</summary>
    public static (int Height, int Width, int OffsetH, int OffsetW) CenterCrop(int height, int width)
    {
        int lo = Math.Min(height, width), hi = Math.Max(height, width);
        if (hi <= MaxAspect * lo) return (height, width, 0, 0);
        int keep = Math.Max(SizeMultiple, (int)(MaxAspect * lo) / SizeMultiple * SizeMultiple);
        int off = (hi - keep) / 2;
        return width > height ? (height, keep, 0, off) : (keep, width, off, 0);
    }

    /// <summary>Upstream <c>crop_angles_to_full</c>: the angles of a centred crop of <paramref name="cropH"/> x <paramref name="cropW"/> source pixels converted to
    /// the full frame's at the same focal length.</summary>
    public static DepthPro2FieldOfView CropAnglesToFull(DepthPro2FieldOfView crop, double cropH, double cropW, double height, double width)
    {
        double fx = cropW / 2 / Math.Tan(crop.HorizontalDegrees * Math.PI / 360.0);
        double fy = cropH / 2 / Math.Tan(crop.VerticalDegrees * Math.PI / 360.0);
        // Upstream computes in float32 at least and returns the input dtype (float32), so the angles are rounded to float like its output.
        return new DepthPro2FieldOfView((float)(2 * Math.Atan(width / 2 / fx) * 180.0 / Math.PI), (float)(2 * Math.Atan(height / 2 / fy) * 180.0 / Math.PI));
    }

    /// <summary>The four tapped trunk outputs concatenated on channels: <c>[1, 5 + patches, 4·384]</c>.</summary>
    public Tensor EncodeTaps(IBackend backend, Tensor input)
    {
        Tensor[] taps = _trunk.EncodeIntermediates(backend, input, TappedBlocks, PrefixResetBlocks);
        try
        {
            int tokens = (int)taps[0].Shape[1], hidden = _trunkPreset.HiddenSize, width = hidden * taps.Length;
            Tensor joined = new(new TensorShape(1, tokens, width), DType.F32);
            Span<float> dst = joined.AsSpan<float>();
            for (int t = 0; t < taps.Length; t++)
            {
                ReadOnlySpan<float> src = taps[t].AsReadOnlySpan<float>();
                for (int r = 0; r < tokens; r++) src.Slice(r * hidden, hidden).CopyTo(dst.Slice(r * width + t * hidden, hidden));
            }

            return joined;
        }
        finally
        {
            foreach (Tensor tap in taps) tap.Dispose();
        }
    }

    // F.interpolate(bilinear, antialias=True) per channel to the full target, then the centre crop.
    private static Tensor ResizeAndCrop(Tensor pixels, int fullH, int fullW, int cropH, int cropW, int offH, int offW)
    {
        int height = (int)pixels.Shape[2], width = (int)pixels.Shape[3];
        float[] resized = new float[fullH * fullW];
        Tensor output = new(new TensorShape(1, 3, cropH, cropW), DType.F32);
        ReadOnlySpan<float> src = pixels.AsReadOnlySpan<float>();
        Span<float> dst = output.AsSpan<float>();
        for (int c = 0; c < 3; c++)
        {
            Resample.BilinearPlaneAntialias(src.Slice(c * height * width, height * width), width, height, resized, fullW, fullH);
            for (int y = 0; y < cropH; y++)
                resized.AsSpan((offH + y) * fullW + offW, cropW).CopyTo(dst.Slice((c * cropH + y) * cropW, cropW));
        }

        return output;
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));
}
