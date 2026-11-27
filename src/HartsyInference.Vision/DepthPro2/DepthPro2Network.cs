using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.CrossFrame;
using HartsyInference.Vision.Dinov3;

namespace HartsyInference.Vision.DepthPro2;

/// <summary>
/// The Depth Pro 2 depth network for one image: the timm DINOv3 backbone of <c>encoder.*</c> (prefix tokens restored after each cross-frame block, as a lone
/// frame does upstream) and <see cref="DepthPro2Head"/>. Input is ImageNet-normalised pixels at their native size; the network replicate-pads the bottom and
/// right edges to a multiple of 32 and crops the inverse depth back. F32 weights are borrowed from the caller; disposing frees the
/// copies the backbone made of 16-bit ones.
/// </summary>
public sealed class DepthPro2Network : IDisposable
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
        try
        {
            _head = new DepthPro2Head(preset.DimsReassemble, preset.DimDecoder, preset.Backbone.HiddenSize, weights);
        }
        catch
        {
            _encoder.Dispose();
            throw;
        }
    }

    public void Dispose() => _encoder.Dispose();

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
        using ExactGemmScope exact = new(backend);
        using Tensor? padded = PadToMultiple(backend, pixels);
        Tensor full = InverseDepthOfPadded(backend, padded ?? pixels, observe);
        if (padded is null) return full;
        using (full) return Crop(backend, full, height, width);
    }

    /// <summary>
    /// Focal-normalised inverse depth <c>[T, 1, H, W]</c> for the normalised frames <c>[T, 3, H, W]</c> of one clip or the views of one scene
    /// (upstream <c>predict_inverse_depth</c> on a clip): the frames attend to each other in the cross-frame blocks, so the result is equivariant
    /// to their order but not per-frame independent. With a <paramref name="ring"/> the input is one frame that attends to the frames the ring
    /// holds as well, and is then stored in it (unbuffered video). A <paramref name="rope"/> cache keeps the backbone's RoPE tables for the next call.
    /// </summary>
    public Tensor PredictInverseDepthFrames(IBackend backend, Tensor frames, CrossFrameKvRing? ring = null, Dinov3RopeCache? rope = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(frames);
        if (frames.DType != DType.F32 || frames.Shape.Rank != 4 || frames.Shape[0] < 1 || frames.Shape[1] != 3
            || frames.Shape[2] < 1 || frames.Shape[3] < 1)
            throw new ArgumentException($"frames must be F32 [T, 3, H, W]; got {frames.DType} {frames.Shape}.", nameof(frames));
        int height = (int)frames.Shape[2], width = (int)frames.Shape[3];
        using ExactGemmScope exact = new(backend);
        using Tensor? padded = PadToMultiple(backend, frames);
        Tensor full = InverseDepthOfPadded(backend, padded ?? frames, null, ring, rope);
        if (padded is null) return full;
        using (full) return Crop(backend, full, height, width);
    }

    /// <summary>The tapped token sequences (<c>[1, 5 + patches, width]</c> each, in <see cref="DepthPro2Preset.ReassembledBlocks"/> order) for pixels whose sides
    /// are multiples of 32; the caller owns them.</summary>
    public Tensor[] EncodeTaps(IBackend backend, Tensor paddedPixels)
    {
        ArgumentNullException.ThrowIfNull(backend);
        using ExactGemmScope exact = new(backend);
        return _encoder.EncodeIntermediates(backend, paddedPixels, _preset.ReassembledBlocks, _preset.CrossFrameBlocks);
    }

    private Tensor InverseDepthOfPadded(IBackend backend, Tensor padded, Action<string, Tensor>? observe, CrossFrameKvRing? ring = null,
        Dinov3RopeCache? rope = null)
    {
        int patch = _preset.Backbone.PatchSize, prefix = 1 + _preset.Backbone.NumRegisterTokens;
        int frames = (int)padded.Shape[0], gridH = (int)padded.Shape[2] / patch, gridW = (int)padded.Shape[3] / patch;
        Tensor[] taps = frames == 1 && ring is null ? EncodeTaps(backend, padded)
            : _encoder.EncodeFrameIntermediates(backend, padded, _preset.ReassembledBlocks, _preset.CrossFrameBlocks, ring, rope);
        try
        {
            if (frames == 1) return _head.Forward(backend, taps, prefix, gridH, gridW, observe);
            Tensor result = new(new TensorShape(frames, 1, padded.Shape[2], padded.Shape[3]), DType.F32);
            Tensor[] frameTaps = new Tensor[taps.Length];
            try
            {
                for (int t = 0; t < frames; t++)
                {
                    try
                    {
                        for (int i = 0; i < taps.Length; i++) frameTaps[i] = FrameOf(backend, taps[i], t);
                        using Tensor map = _head.Forward(backend, frameTaps, prefix, gridH, gridW, observe);
                        // The first write covers the whole result, so a GPU backend allocates it on the device instead of uploading a host buffer.
                        if (t == 0) StridedCopies.FillFrom(backend, result, map);
                        StridedCopies.ScatterAxis(backend, result, map, 0, t);
                    }
                    finally
                    {
                        foreach (Tensor tap in frameTaps) tap?.Dispose();
                        Array.Clear(frameTaps);
                    }
                }

                return result;
            }
            catch
            {
                result.Dispose();
                throw;
            }
        }
        finally
        {
            foreach (Tensor tap in taps) tap.Dispose();
        }
    }

    // Frame t of [T, S, C] as a new [1, S, C].
    private static Tensor FrameOf(IBackend backend, Tensor tokens, int t)
    {
        Tensor frame = new(new TensorShape(1, tokens.Shape[1], tokens.Shape[2]), DType.F32);
        StridedCopies.SliceAxis(backend, frame, tokens, 0, t, 1);
        return frame;
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

    private static Tensor Crop(IBackend backend, Tensor full, int height, int width)
    {
        Tensor cropped = new(new TensorShape(full.Shape[0], 1, height, width), DType.F32);
        StridedView window = StridedView.Of(full.Shape).Slice(2, 0, height).Slice(3, 0, width);
        backend.CopyStrided(cropped, full, StridedCopySpec.Gather(window));
        return cropped;
    }
}
