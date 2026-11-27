using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Depthwise convolution along the frame axis of token-major data; the host reference is the default.
public partial interface IBackend
{
    /// <summary>Depthwise 1-D convolution over the frame axis of token-major <c>[frames, spatial, channels]</c> data:
    /// <c>output[t, s, c] = Σ_j weight[c, j] · input[t + j − pad, s, c]</c>, zero outside the clip. No transpose: each channel
    /// is convolved in place across frames.</summary>
    /// <param name="output">F32 <c>[frames, spatial, channels]</c>.</param>
    /// <param name="input">F32 <c>[frames, spatial, channels]</c>.</param>
    /// <param name="weight">F32 taps <c>[channels, taps]</c> (any shape of that size).</param>
    void DepthwiseConvFrames(Tensor output, Tensor input, Tensor weight, int frames, int spatial, int channels, int taps, int pad) =>
        DepthwiseConvFramesReference.Run(output, input, weight, frames, spatial, channels, taps, pad);
}
