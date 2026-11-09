using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Streaming;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>What one SANA-WM block carries from chunk to chunk in a chunk-causal stream (upstream's per-block
/// <c>kv_cache</c> slots). A Gated-DeltaNet block carries recurrent states and the input tails of its key convolutions; a
/// softmax block carries windows of keys and values; every block carries the input tail of its FFN's temporal convolution.</summary>
/// <remarks>States are in this engine's layouts: a scan state is <c>[H, Dk, Dv]</c> (key-major), which is the transpose of
/// upstream's <c>S_kv</c> slot and equal to its camera slot; keys and values are head-major <c>[1, H, N, D]</c>.
/// <para>Storage dtypes follow upstream's cache, which holds its keys, values and key-convolution inputs in BF16: the softmax
/// windows and the two key tails are stored as <see cref="StorageDType"/> and widened where they are read; scan states stay
/// F32 (they are small and recur), and the FFN tail keeps its own dtype. A pass that closes a chunk converts only what it
/// stages, so the F32 chunk tensors are released as soon as the narrow copies exist.</para></remarks>
internal sealed class SanaWmBlockStream
{
    /// <summary>Earlier frames a key convolution reads: one less than its taps.</summary>
    internal const int KeyTailFrames = SanaWmTemporalConv.SourceTaps - 1;

    /// <summary>Earlier frames the FFN's temporal convolution reads.</summary>
    internal const int FfnTailFrames = 1;

    /// <summary>The dtype softmax windows and key tails are held in between chunks.</summary>
    internal static readonly DType StorageDType = DType.BF16;

    private const int TokenAxis = 2;
    private const int KeyValueStreams = 2;

    internal SanaWmBlockStream(IBackend backend, StreamingStateSet states, bool softmax, bool camera, int spatial,
        TokenWindowPolicy softmaxWindow)
    {
        FfnTail = states.Add(new FrameTailCache(backend, FfnTailFrames, 0, spatial));
        if (softmax)
        {
            MainWindow = states.Add(new RollingTokenWindow(backend, softmaxWindow, KeyValueStreams, TokenAxis, StorageDType));
            if (camera)
                CameraWindow = states.Add(new RollingTokenWindow(backend, softmaxWindow, KeyValueStreams, TokenAxis, StorageDType));
            return;
        }

        Numerator = states.Add(new StreamingTensorSlot(backend));
        Denominator = states.Add(new StreamingTensorSlot(backend));
        KeyTail = states.Add(new FrameTailCache(backend, KeyTailFrames, 0, spatial, StorageDType));
        if (camera)
        {
            CameraNumerator = states.Add(new StreamingTensorSlot(backend));
            CameraKeyTail = states.Add(new FrameTailCache(backend, KeyTailFrames, 0, spatial, StorageDType));
        }
    }

    /// <summary>GDN main branch: the numerator scan state <c>[H, D, D]</c>.</summary>
    internal StreamingTensorSlot? Numerator { get; }

    /// <summary>GDN main branch: the denominator scan state <c>[H, D, 1]</c>.</summary>
    internal StreamingTensorSlot? Denominator { get; }

    /// <summary>GDN main branch: the raw key projection of the newest frames, token rows, in <see cref="StorageDType"/>.</summary>
    internal FrameTailCache? KeyTail { get; }

    /// <summary>GDN camera branch: the numerator scan state <c>[H, D, D]</c>.</summary>
    internal StreamingTensorSlot? CameraNumerator { get; }

    /// <summary>GDN camera branch: the raw key projection of the newest frames, token rows, in <see cref="StorageDType"/>.</summary>
    internal FrameTailCache? CameraKeyTail { get; }

    /// <summary>Softmax main branch: keys after RoPE and values, stream 0 and 1, in <see cref="StorageDType"/>.</summary>
    internal RollingTokenWindow? MainWindow { get; }

    /// <summary>Softmax camera branch: keys and values after the ray transform, stream 0 and 1, in <see cref="StorageDType"/>.</summary>
    internal RollingTokenWindow? CameraWindow { get; }

    /// <summary>The FFN point-convolution output of the newest frame, token rows in the FFN dtype.</summary>
    internal FrameTailCache FfnTail { get; }
}
