using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Dinov3;

namespace HartsyInference.Vision.DepthPro2;

/// <summary>
/// The Depth Pro 2 depth network for one image: the timm DINOv3 backbone of <c>encoder.*</c> (prefix tokens restored after each cross-frame block, as a lone
/// frame does upstream) and <see cref="DepthPro2Head"/>. Input is ImageNet-normalised pixels at their native size; the network replicate-pads the bottom and
/// right edges to a multiple of 32 and crops the inverse depth back. Weights are borrowed from the caller, F32.
/// </summary>
public sealed class DepthPro2Network
{
    private readonly DepthPro2Preset _preset;
    private readonly Dinov3VisionEncoder _encoder;
    private readonly DepthPro2Head _head;

    /// <param name="weights">The checkpoint's tensors by their upstream names (an <c>apple/DepthPro2-*</c> <c>model.safetensors</c>).</param>
    /// <exception cref="InvalidOperationException">A weight is missing or has the wrong shape.</exception>
    public DepthPro2Network(DepthPro2Preset preset, IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(preset);
        ArgumentNullException.ThrowIfNull(weights);
        _preset = preset;
        _encoder = new Dinov3VisionEncoder(preset.Backbone);
        _encoder.LoadWeights(Dinov3TimmKeys.ToHuggingFace(weights, "encoder.", preset.Backbone));
        _head = new DepthPro2Head(preset.DimsReassemble, preset.DimDecoder, preset.Backbone.HiddenSize, weights);
    }

    public DepthPro2Preset Preset => _preset;

    public Dinov3VisionEncoder Encoder => _encoder;

    public DepthPro2Head Head => _head;

    /// <summary>The backbone's and the head's tensors.</summary>
    public IEnumerable<Tensor> EnumerateWeights() => _encoder.EnumerateWeights().Concat(_head.EnumerateWeights());

    /// <summary>Focal-normalised inverse depth <c>[1, 1, H, W]</c> for normalised pixels <c>[1, 3, H, W]</c>; zero where infinitely far (sky). Metric depth is
    /// <c>f_px / (W · inverseDepth)</c>. <paramref name="observe"/> sees each head stage, see <see cref="DepthPro2Head.Forward"/>.</summary>
    public Tensor PredictInverseDepth(IBackend backend, Tensor pixels, Action<string, Tensor>? observe = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(pixels);
        if (pixels.DType != DType.F32 || pixels.Shape.Rank != 4 || pixels.Shape[0] != 1 || pixels.Shape[1] != 3 || pixels.Shape[2] < 1 || pixels.Shape[3] < 1)
            throw new ArgumentException($"pixels must be F32 [1, 3, H, W]; got {pixels.DType} {pixels.Shape}.", nameof(pixels));
        int height = (int)pixels.Shape[2], width = (int)pixels.Shape[3];
        using Tensor? padded = PadToMultiple(backend, pixels);
        Tensor full = InverseDepthOfPadded(backend, padded ?? pixels, observe);
        if (padded is null) return full;
        using (full) return Crop(full, height, width);
    }

    /// <summary>The tapped token sequences (<c>[1, 5 + patches, width]</c> each, in <see cref="DepthPro2Preset.ReassembledBlocks"/> order) for pixels whose sides
    /// are multiples of 32; the caller owns them.</summary>
    public Tensor[] EncodeTaps(IBackend backend, Tensor paddedPixels) =>
        _encoder.EncodeIntermediates(backend, paddedPixels, _preset.ReassembledBlocks, _preset.CrossFrameBlocks);

    private Tensor InverseDepthOfPadded(IBackend backend, Tensor padded, Action<string, Tensor>? observe)
    {
        int patch = _preset.Backbone.PatchSize;
        Tensor[] taps = EncodeTaps(backend, padded);
        try
        {
            return _head.Forward(backend, taps, 1 + _preset.Backbone.NumRegisterTokens, (int)padded.Shape[2] / patch, (int)padded.Shape[3] / patch, observe);
        }
        finally
        {
            foreach (Tensor tap in taps) tap.Dispose();
        }
    }

    /// <summary>The pixels replicate-padded at the bottom and right to multiples of <see cref="DepthPro2Preset.InputMultiple"/>, or null when they already are.</summary>
    public static Tensor? PadToMultiple(IBackend backend, Tensor pixels)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(pixels);
        long h = pixels.Shape[2], w = pixels.Shape[3];
        int padH = (int)(-h % DepthPro2Preset.InputMultiple + DepthPro2Preset.InputMultiple) % DepthPro2Preset.InputMultiple;
        int padW = (int)(-w % DepthPro2Preset.InputMultiple + DepthPro2Preset.InputMultiple) % DepthPro2Preset.InputMultiple;
        if (padH == 0 && padW == 0) return null;
        Tensor padded = new(new TensorShape(pixels.Shape[0], pixels.Shape[1], h + padH, w + padW), DType.F32);
        backend.Pad(padded, pixels, new PadSpec(PadMode.Replicate, [0, 0, 0, 0], [0, 0, padH, padW]));
        return padded;
    }

    private static Tensor Crop(Tensor full, int height, int width)
    {
        int fullW = (int)full.Shape[3];
        Tensor cropped = new(new TensorShape(1, 1, height, width), DType.F32);
        ReadOnlySpan<float> src = full.AsReadOnlySpan<float>();
        Span<float> dst = cropped.AsSpan<float>();
        for (int y = 0; y < height; y++)
            src.Slice(y * fullW, width).CopyTo(dst.Slice(y * width, width));
        return cropped;
    }
}
