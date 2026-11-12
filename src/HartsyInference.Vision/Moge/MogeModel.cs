using HartsyInference.Core.Backends;
using HartsyInference.Core.Imaging;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.Vision.Moge;

/// <summary>What <see cref="MogeModel.Infer"/> produced for one image, all at the input's size.</summary>
/// <param name="Points">Camera-space points, planar <c>[3, height, width]</c>, x right, y down, z forward (the OpenCV convention); infinite where the mask is off and masking was applied.</param>
/// <param name="Depth"><c>height · width</c> depths (<c>z</c> after the shift); infinite where masked.</param>
/// <param name="Mask"><c>height · width</c> validity flags (<c>mask &gt; threshold</c>).</param>
/// <param name="Focal">The focal length relative to the half image diagonal.</param>
/// <param name="Shift">The z shift added to the raw output.</param>
/// <param name="Fx">The horizontal focal length in units of the image width (the intrinsics matrix's <c>fx</c>).</param>
/// <param name="Fy">The vertical focal length in units of the image height.</param>
public sealed record MogeResult(float[] Points, float[] Depth, bool[] Mask, double Focal, double Shift, double Fx, double Fy);

/// <summary>
/// MoGe v1 (<c>microsoft/MoGe</c>, MIT; weights <c>Ruicheng/moge-vitl</c>): monocular geometry estimation. An image is resized to a token budget, a DINOv2-L/14 backbone (no registers)
/// gives the last four blocks after the final norm, and <see cref="MogeHead"/> turns them into a point map and a mask; <see cref="Infer"/> then recovers the shift and focal length
/// (<see cref="MogeFocalShift"/>) and returns camera-space points. The backbone and head run in F32 (upstream's <c>infer</c> defaults to fp16 autocast). The weights are borrowed from the caller.
/// </summary>
public sealed class MogeModel
{
    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];
    private const int Patch = 14;

    private readonly MogeConfig _config;
    private readonly Dinov2VisionEncoder _backbone;
    private readonly MogeHead _head;
    private readonly int[] _layers;

    /// <param name="config">The architecture knobs (<see cref="MogeConfig.FromJson"/>).</param>
    /// <param name="weights">The checkpoint's tensors by name: <c>backbone.*</c> (hub DINOv2 names) and <c>head.*</c>, F32; they must outlive this object.</param>
    /// <exception cref="InvalidOperationException">A weight is missing, not F32 or of the wrong shape.</exception>
    public MogeModel(MogeConfig config, IReadOnlyDictionary<string, Tensor> weights)
        : this(config, Dinov2Preset.Large, weights)
    {
    }

    /// <summary>The same network on another backbone size, for tests; the released one is <see cref="Dinov2Preset.Large"/>.</summary>
    internal MogeModel(MogeConfig config, Dinov2Preset preset, IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(preset);
        ArgumentNullException.ThrowIfNull(weights);
        if (config.IntermediateLayers > preset.NumLayers)
            throw new ArgumentException($"The head taps {config.IntermediateLayers} blocks but the backbone has {preset.NumLayers}.", nameof(config));
        _config = config;
        _backbone = new Dinov2VisionEncoder(preset);
        _backbone.LoadWeights(Dinov2HubKeys.Remap(weights, "backbone.", preset, requireF32: true));
        _head = new MogeHead(config, weights, "head.", preset.HiddenSize);
        _layers = [.. Enumerable.Range(preset.NumLayers - config.IntermediateLayers, config.IntermediateLayers)];
    }

    /// <summary>
    /// The tensors the model reads, for <c>IBackend.PreloadWeights</c> and <c>FreeWeights</c>: the backbone's (its query, key and value projections are views of the fused checkpoint tensors, and
    /// the backend caches by tensor object, so the checkpoint's own fused tensors are not the ones to preload or free) and the head's.
    /// </summary>
    public IEnumerable<Tensor> EnumerateWeights() => _backbone.EnumerateWeights().Concat(_head.EnumerateWeights());

    /// <summary>
    /// The model's <c>forward</c>: resize to the token budget, run backbone and head, resize the outputs to the input, remap the points. Returns planar points <c>[3, height, width]</c>
    /// and the raw mask <c>[height, width]</c>.
    /// </summary>
    /// <param name="rgb">Planar RGB <c>[3, height, width]</c> in 0 to 1.</param>
    /// <param name="numTokens">The token budget; the image is resized to about <c>numTokens · 14²</c> pixels.</param>
    public (float[] Points, float[] Mask) Forward(IBackend backend, ReadOnlySpan<float> rgb, int height, int width, int numTokens)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (height <= 0 || width <= 0 || rgb.Length != 3 * height * width)
            throw new ArgumentException($"An image of 3 x {height} x {width} holds {3 * height * width} values; got {rgb.Length}.", nameof(rgb));
        double factor = Math.Sqrt((double)numTokens * Patch * Patch / ((double)height * width));
        int resizedWidth = (int)(width * factor), resizedHeight = (int)(height * factor);
        int gridH = resizedHeight / Patch, gridW = resizedWidth / Patch;
        if (gridH < 1 || gridW < 1)
            throw new ArgumentException($"A token budget of {numTokens} leaves a {gridH} x {gridW} grid for a {height} x {width} image.", nameof(numTokens));

        using Tensor pixels = Preprocess(rgb, height, width, resizedHeight, resizedWidth, gridH * Patch, gridW * Patch);
        Tensor[] sequences = _backbone.EncodeIntermediates(backend, pixels, _layers);
        List<Tensor> views = [];
        try
        {
            // Drop the class token of every tap: [1, 1 + n, c] to a [1, n, c] view of the same bytes.
            foreach (Tensor sequence in sequences)
            {
                using Tensor rows = sequence.Reshape(new TensorShape(sequence.Shape[1], sequence.Shape[2]));
                using Tensor patches = rows.SliceRows(1, sequence.Shape[1] - 1);
                views.Add(patches.Reshape(new TensorShape(1, patches.Shape[0], patches.Shape[1])));
            }

            (Tensor rawPoints, Tensor rawMask) = _head.Forward(backend, views, resizedHeight, resizedWidth, Patch);
            using (rawPoints)
            using (rawMask)
            {
                float[] points = Resize(backend, rawPoints, height, width);
                float[] mask = Resize(backend, rawMask, height, width);
                Remap(points, height * width);
                return (points, mask);
            }
        }
        finally
        {
            foreach (Tensor view in views)
                view.Dispose();
            foreach (Tensor sequence in sequences)
                sequence.Dispose();
        }
    }

    /// <summary>Bicubic antialiased resize to the token budget, ImageNet normalisation, then bilinear antialiased resize to a multiple of 14.</summary>
    private static Tensor Preprocess(ReadOnlySpan<float> rgb, int height, int width, int resizedHeight, int resizedWidth, int cropHeight, int cropWidth)
    {
        Tensor pixels = new(new TensorShape(1, 3, cropHeight, cropWidth), DType.F32);
        Span<float> target = pixels.AsSpan<float>();
        float[] resized = new float[resizedHeight * resizedWidth];
        for (int c = 0; c < 3; c++)
        {
            Resample.BicubicPlane(rgb.Slice(c * height * width, height * width), width, height, resized, resizedWidth, resizedHeight, -0.5f, antialias: true);
            float mean = Mean[c], inverse = 1f / Std[c];
            for (int i = 0; i < resized.Length; i++)
                resized[i] = (resized[i] - mean) * inverse;
            Resample.BilinearPlaneAntialias(resized, resizedWidth, resizedHeight, target.Slice(c * cropHeight * cropWidth, cropHeight * cropWidth), cropWidth, cropHeight);
        }

        return pixels;
    }

    /// <summary>A <c>[1, c, h, w]</c> head output bilinearly resized (half-pixel centres, no antialias) to the image, as planar floats.</summary>
    private static float[] Resize(IBackend backend, Tensor output, int height, int width)
    {
        using Tensor resized = new(new TensorShape(1, output.Shape[1], height, width), DType.F32);
        backend.InterpolateBilinear2D(resized, output, alignCorners: false);
        return resized.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary><c>_remap_points</c> on planar points.</summary>
    private void Remap(float[] points, int plane)
    {
        switch (_config.Remap)
        {
            case MogeRemap.Linear:
                break;
            case MogeRemap.Sinh:
                for (int i = 0; i < points.Length; i++)
                    points[i] = MathF.Sinh(points[i]);
                break;
            case MogeRemap.Exp:
                for (int i = 0; i < plane; i++)
                {
                    float z = MathF.Exp(points[2 * plane + i]);
                    points[i] *= z;
                    points[plane + i] *= z;
                    points[2 * plane + i] = z;
                }

                break;
            case MogeRemap.SinhExp:
                for (int i = 0; i < plane; i++)
                {
                    points[i] = MathF.Sinh(points[i]);
                    points[plane + i] = MathF.Sinh(points[plane + i]);
                    points[2 * plane + i] = MathF.Exp(points[2 * plane + i]);
                }

                break;
        }
    }

    /// <summary>
    /// The user-facing <c>infer</c> with <c>force_projection=False</c> (what SAM 3D calls): <see cref="Forward"/>, the focal length and shift (from <paramref name="fovXDegrees"/> when known,
    /// else both solved), the shift added to <c>z</c>, and the points of masked-out pixels set to infinity.
    /// </summary>
    /// <param name="rgb">Planar RGB <c>[3, height, width]</c> in 0 to 1.</param>
    /// <param name="resolutionLevel">0 to 9: where the token budget sits between the config's minimum and maximum.</param>
    /// <param name="fovXDegrees">The horizontal field of view when known.</param>
    /// <param name="applyMask">Whether to set the points and depths of pixels outside the mask to infinity.</param>
    public MogeResult Infer(IBackend backend, ReadOnlySpan<float> rgb, int height, int width, int resolutionLevel = 9, double? fovXDegrees = null, bool applyMask = true)
    {
        if (resolutionLevel is < 0 or > 9)
            throw new ArgumentOutOfRangeException(nameof(resolutionLevel), resolutionLevel, "The resolution level is 0 to 9.");
        (int minTokens, int maxTokens) = _config.NumTokensRange;
        int numTokens = (int)(minTokens + resolutionLevel / 9.0 * (maxTokens - minTokens));
        (float[] points, float[] rawMask) = Forward(backend, rgb, height, width, numTokens);

        bool[] mask = new bool[height * width];
        for (int i = 0; i < mask.Length; i++)
            mask[i] = rawMask[i] > _config.MaskThreshold;
        double aspect = (double)width / height;
        double? knownFocal = fovXDegrees is double fov ? aspect / Math.Sqrt(1 + aspect * aspect) / Math.Tan(fov * Math.PI / 360.0) : null;
        (double focal, double shift) = MogeFocalShift.Recover(points, mask, height, width, knownFocal);

        int plane = height * width;
        float[] depth = new float[plane];
        for (int i = 0; i < plane; i++)
        {
            points[2 * plane + i] += (float)shift;
            depth[i] = points[2 * plane + i];
        }

        if (applyMask)
        {
            for (int i = 0; i < plane; i++)
            {
                if (mask[i])
                    continue;
                points[i] = points[plane + i] = points[2 * plane + i] = float.PositiveInfinity;
                depth[i] = float.PositiveInfinity;
            }
        }

        double root = Math.Sqrt(1 + aspect * aspect);
        return new MogeResult(points, depth, mask, focal, shift, focal / 2 * root / aspect, focal / 2 * root);
    }
}
