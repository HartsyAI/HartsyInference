using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Turns the avatar's raw audio windows into the per-frame audio tokens the DiT's blocks read (the reference's <c>forward</c> audio path,
/// <c>longcat_video_dit_avatar.py</c> @ 58eb336).</summary>
/// <remarks>
/// <c>rawAudio</c> is <c>[1 + vaeScale·(frames−1), window·block·channel]</c> F32: one row per audio frame, each row flattened in (window, block,
/// channel) order. The first latent frame takes row 0 as its window. Each later latent frame takes the <c>vaeScale</c> rows that follow, and
/// from them the reference's slot selection: the first sub-frame gives windows 0..middle, the middle sub-frames give window middle, and the
/// last sub-frame gives windows middle..window−1 (middle = window / 2). The slots are stacked per frame, so the projection's latter input is
/// <c>[frames−1, (window + vaeScale − 1)·block·channel]</c>. The projection then gives <c>[frames·contextTokens, outputDim]</c>.
/// </remarks>
public static class LongCatAvatarAudio
{
    /// <summary>Projects one clip's raw audio windows to the per-frame audio tokens. The raw tensor is read on the host (F32, CPU), since its
    /// rows are gathered per slot; it is small (one row per audio frame).</summary>
    public static Tensor FrameTokens(IBackend backend, LongCatAudioProjection projection, Tensor rawAudio, int frames, int window, int block,
        int channel, int vaeScale)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(projection);
        ArgumentNullException.ThrowIfNull(rawAudio);
        if (frames < 2)
            throw new ArgumentOutOfRangeException(nameof(frames), frames, "The avatar audio path needs at least two latent frames (a first frame and one latter frame).");
        if (rawAudio.DType != DType.F32)
            throw new ArgumentException($"The raw audio must be F32, got {rawAudio.DType}.", nameof(rawAudio));
        int rowWidth = window * block * channel, audioRows = 1 + vaeScale * (frames - 1);
        if (rawAudio.Shape.Rank != 2 || rawAudio.Shape[0] != audioRows || rawAudio.Shape[1] != rowWidth)
            throw new ArgumentException($"The raw audio must be [{audioRows}, {rowWidth}] for {frames} latent frames, got {rawAudio.Shape}.", nameof(rawAudio));

        int middle = window / 2, slotWidth = block * channel;
        int slots = (middle + 1) + Math.Max(0, vaeScale - 2) + (window - middle);
        if (slots != window + vaeScale - 1)
            throw new InvalidOperationException("The slot count must equal window + vaeScale − 1 (the reference's seqLenVf).");
        ReadOnlySpan<float> source = rawAudio.AsReadOnlySpan<float>();

        Tensor first = new(new TensorShape(1, rowWidth), DType.F32);
        source[..rowWidth].CopyTo(first.AsSpan<float>());

        int latter = frames - 1;
        Tensor latterTensor = new(new TensorShape(latter, slots * slotWidth), DType.F32);
        Span<float> destination = latterTensor.AsSpan<float>();
        for (int f = 0; f < latter; f++)
        {
            int rowBase = 1 + vaeScale * f;
            int slot = 0;
            // First sub-frame: windows 0..middle.
            for (int w = 0; w <= middle; w++, slot++)
                CopySlot(source, destination, rowBase, w, slot, slotWidth, rowWidth, f, slots);
            // Middle sub-frames: window middle.
            for (int n = 1; n <= vaeScale - 2; n++, slot++)
                CopySlot(source, destination, rowBase + n, middle, slot, slotWidth, rowWidth, f, slots);
            // Last sub-frame: windows middle..window-1.
            for (int w = middle; w < window; w++, slot++)
                CopySlot(source, destination, rowBase + vaeScale - 1, w, slot, slotWidth, rowWidth, f, slots);
        }

        try
        {
            return projection.Forward(backend, first, latterTensor);
        }
        finally
        {
            first.Dispose();
            latterTensor.Dispose();
        }
    }

    private static void CopySlot(ReadOnlySpan<float> source, Span<float> destination, int row, int window, int slot, int slotWidth, int rowWidth,
        int frame, int slots)
    {
        // The row is (window, block, channel) flattened: window w starts at w·block·channel, and a slot is one window's block·channel values.
        int from = row * rowWidth + window * slotWidth;
        int to = frame * slots * slotWidth + slot * slotWidth;
        source.Slice(from, slotWidth).CopyTo(destination.Slice(to, slotWidth));
    }
}
