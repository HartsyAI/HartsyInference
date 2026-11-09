using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Imaging;

/// <summary>One RGB frame as interleaved 8-bit pixels (HWC), the layout PNG decoders and ffmpeg produce.</summary>
public readonly record struct RawRgbFrame(ReadOnlyMemory<byte> Rgb, int Width, int Height)
{
    /// <summary>Throws when the buffer does not hold <c>Width·Height·3</c> bytes.</summary>
    public void Validate()
    {
        if (Width <= 0 || Height <= 0 || Rgb.Length != (long)Width * Height * 3)
            throw new ArgumentException($"An RGB frame of {Width}x{Height} needs {(long)Width * Height * 3} bytes, got {Rgb.Length}.");
    }
}

/// <summary>A ViT's patch geometry: square patches, a square spatial merge, and the frames one temporal patch spans.</summary>
public readonly record struct VisionPatchSpec(int PatchSize, int MergeSize, int TemporalPatchSize, int Channels = 3)
{
    /// <summary>The side multiple a resized image must have: one merge block of patches.</summary>
    public int Factor => PatchSize * MergeSize;

    /// <summary>The length of one patch row: <c>channels · temporal · patch²</c>.</summary>
    public int FeatureDim => Channels * TemporalPatchSize * PatchSize * PatchSize;
}

/// <summary>Patch rows for one vision block and its grid, in patch units. Owns <see cref="PixelValues"/>.</summary>
/// <param name="PixelValues"><c>[GridT·GridH·GridW, FeatureDim]</c> F32 rows in merge-block order.</param>
public sealed record VisionPatchGrid(Tensor PixelValues, int GridT, int GridH, int GridW, int MergeSize) : IDisposable
{
    /// <summary>The grid height after the spatial merge.</summary>
    public int MergedHeight => GridH / MergeSize;

    /// <summary>The grid width after the spatial merge.</summary>
    public int MergedWidth => GridW / MergeSize;

    /// <inheritdoc/>
    public void Dispose() => PixelValues.Dispose();
}

/// <summary>Where an 8-bit resize rounds back onto the 0–255 grid.</summary>
public enum U8Rounding
{
    /// <summary>Never: the resized values stay fractional.</summary>
    None,

    /// <summary>After each separable pass, as PIL does (its horizontal pass writes an 8-bit intermediate).</summary>
    EachPass,

    /// <summary>Once, after both passes (half-to-even, then clamped): a float resize whose result is stored as 8-bit.</summary>
    Final,
}

/// <summary>Which separable filter resizes a frame, with the options that decide parity with a given reference.</summary>
/// <param name="Bicubic">Keys bicubic (a = −0.5, PIL's) when true; triangle (bilinear) when false.</param>
/// <param name="Antialias">Stretches the kernel on a downscale (PIL, torchvision <c>antialias=True</c>).</param>
/// <param name="Rounding">Where the 8-bit result is rounded.</param>
public readonly record struct ResampleOptions(bool Bicubic, bool Antialias, U8Rounding Rounding)
{
    /// <summary>PIL <c>BICUBIC</c> on an 8-bit image (rounded after each pass). Hugging Face's fast Qwen-VL processors (torchvision on a
    /// uint8 tensor) match it: measured against them, 0.2–0.3% of values differ by one level, where rounding only once leaves 10–20%
    /// off, some by many levels.</summary>
    public static ResampleOptions PilBicubic => new(Bicubic: true, Antialias: true, Rounding: U8Rounding.EachPass);

    /// <summary>PIL <c>BILINEAR</c> on an 8-bit image.</summary>
    public static ResampleOptions PilBilinear => new(Bicubic: false, Antialias: true, Rounding: U8Rounding.EachPass);
}

/// <summary>Turns RGB frames into a ViT's patch rows: resize, rescale to [0, 1], normalize, and flatten patches in
/// <b>merge-block order</b> (each run of <c>merge²</c> rows is one spatial block, the order Qwen-VL mergers and vision rotary
/// tables assume). Model-agnostic: a model supplies its patch geometry, mean/std and filter.</summary>
/// <remarks>Host code, run once per frame before the vision tower; never on a per-token path.</remarks>
public sealed class VisionPatchPreprocessor
{
    private readonly VisionPatchSpec _spec;
    private readonly float[] _mean;
    private readonly float[] _std;
    private readonly ResampleOptions _resample;

    /// <param name="spec">The tower's patch geometry.</param>
    /// <param name="mean">Per-channel mean, applied after rescaling to [0, 1].</param>
    /// <param name="std">Per-channel standard deviation.</param>
    /// <param name="resample">The resize filter.</param>
    public VisionPatchPreprocessor(VisionPatchSpec spec, IReadOnlyList<float> mean, IReadOnlyList<float> std, ResampleOptions resample)
    {
        if (spec.Channels != 3) throw new ArgumentException("Only RGB (3-channel) towers are supported.", nameof(spec));
        if (mean.Count != 3 || std.Count != 3) throw new ArgumentException("Mean and std need one value per RGB channel.");
        _spec = spec;
        _mean = [.. mean];
        _std = [.. std];
        _resample = resample;
    }

    /// <summary>The patch geometry this preprocessor produces.</summary>
    public VisionPatchSpec Spec => _spec;

    /// <summary>Preprocesses one still frame resized to (<paramref name="targetHeight"/>, <paramref name="targetWidth"/>). A tower
    /// whose temporal patch spans several frames gets the still frame repeated across it, as the Qwen-VL processors do.</summary>
    public VisionPatchGrid Preprocess(RawRgbFrame frame, int targetHeight, int targetWidth)
    {
        float[] planes = ResizeNormalize(frame, targetHeight, targetWidth);
        float[][] temporal = new float[_spec.TemporalPatchSize][];
        Array.Fill(temporal, planes);
        return new VisionPatchGrid(Patchify(temporal, targetHeight, targetWidth, _spec), 1, targetHeight / _spec.PatchSize,
            targetWidth / _spec.PatchSize, _spec.MergeSize);
    }

    /// <summary>Preprocesses exactly <see cref="VisionPatchSpec.TemporalPatchSize"/> frames as one temporal patch (grid t = 1).</summary>
    public VisionPatchGrid Preprocess(IReadOnlyList<RawRgbFrame> frames, int targetHeight, int targetWidth)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frames.Count != _spec.TemporalPatchSize)
            throw new ArgumentException($"A temporal patch takes exactly {_spec.TemporalPatchSize} frames, got {frames.Count}.", nameof(frames));
        float[][] temporal = frames.Select(f => ResizeNormalize(f, targetHeight, targetWidth)).ToArray();
        return new VisionPatchGrid(Patchify(temporal, targetHeight, targetWidth, _spec), 1, targetHeight / _spec.PatchSize,
            targetWidth / _spec.PatchSize, _spec.MergeSize);
    }

    /// <summary>Resizes one HWC 8-bit frame and returns normalized CHW planes <c>[3, H, W]</c>.</summary>
    public float[] ResizeNormalize(RawRgbFrame frame, int targetHeight, int targetWidth)
    {
        frame.Validate();
        if (targetHeight % _spec.Factor != 0 || targetWidth % _spec.Factor != 0)
            throw new ArgumentException($"Target {targetHeight}x{targetWidth} is not a multiple of {_spec.Factor} (patch × merge).");
        float[] hwc = new float[(long)targetHeight * targetWidth * 3];
        bool eachPass = _resample.Rounding == U8Rounding.EachPass;
        if (_resample.Bicubic)
            Resample.BicubicHwc8(frame.Rgb.Span, frame.Width, frame.Height, 3, hwc, targetWidth, targetHeight, -0.5f,
                _resample.Antialias, eachPass);
        else
            Resample.BilinearHwc8(frame.Rgb.Span, frame.Width, frame.Height, 3, hwc, targetWidth, targetHeight,
                _resample.Antialias, eachPass);
        if (_resample.Rounding == U8Rounding.Final)
        {
            for (long i = 0; i < hwc.LongLength; i++) hwc[i] = Math.Clamp(MathF.Round(hwc[i], MidpointRounding.ToEven), 0f, 255f);
        }

        long plane = (long)targetHeight * targetWidth;
        float[] chw = new float[3 * plane];
        for (int c = 0; c < 3; c++)
        {
            // The fast processors fuse the rescale into the normalize as (x − 255·mean) / (255·std).
            float shift = 255f * _mean[c];
            float scale = 255f * _std[c];
            for (long i = 0; i < plane; i++) chw[c * plane + i] = (hwc[i * 3 + c] - shift) / scale;
        }
        return chw;
    }

    /// <summary>Flattens normalized CHW planes into merge-block patch rows <c>[gh·gw, C·T·p²]</c>. <paramref name="temporal"/> holds
    /// one <c>[C, H, W]</c> array per frame of the temporal patch.</summary>
    public static unsafe Tensor Patchify(IReadOnlyList<float[]> temporal, int height, int width, VisionPatchSpec spec)
    {
        if (temporal.Count != spec.TemporalPatchSize)
            throw new ArgumentException($"Expected {spec.TemporalPatchSize} temporal planes, got {temporal.Count}.", nameof(temporal));
        int patch = spec.PatchSize, merge = spec.MergeSize, channels = spec.Channels;
        if (height % spec.Factor != 0 || width % spec.Factor != 0)
            throw new ArgumentException($"{height}x{width} is not a multiple of {spec.Factor} (patch × merge).");
        int gridH = height / patch, gridW = width / patch;
        int featDim = spec.FeatureDim;
        long plane = (long)height * width;
        Tensor rows = new(new TensorShape(gridH * gridW, featDim), DType.F32);
        float* dst = (float*)rows.DataPointer;
        long token = 0;
        for (int hb = 0; hb < gridH / merge; hb++)
            for (int wb = 0; wb < gridW / merge; wb++)
                for (int mh = 0; mh < merge; mh++)
                    for (int mw = 0; mw < merge; mw++)
                    {
                        int row0 = (hb * merge + mh) * patch;
                        int col0 = (wb * merge + mw) * patch;
                        float* tokenDst = dst + token * featDim;
                        int idx = 0;
                        for (int c = 0; c < channels; c++)
                            for (int tp = 0; tp < spec.TemporalPatchSize; tp++)
                            {
                                float[] img = temporal[tp];
                                for (int ph = 0; ph < patch; ph++)
                                {
                                    long rowStart = c * plane + (long)(row0 + ph) * width + col0;
                                    for (int pw = 0; pw < patch; pw++) tokenDst[idx++] = img[rowStart + pw];
                                }
                            }
                        token++;
                    }
        return rows;
    }
}
