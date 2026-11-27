using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Channels-last depthwise convolution fused with a gated linear unit; the host reference is the default.
public partial interface IBackend
{
    /// <summary>Depthwise <c>k×k</c> convolution (stride 1, zero padding <c>k/2</c>) over channels-last frames, then a GLU:
    /// with <c>d</c> the convolution output (bias added), <c>output = d[..., :H] · SiLU(d[..., H:])</c>. The convolution's
    /// output is never stored; each output channel reads its value and gate channels and writes their product. With
    /// <paramref name="inputSilu"/> the convolution reads <c>SiLU(input)</c>, so the activation of the GEMM that produced the
    /// input needs no pass of its own.</summary>
    /// <param name="output">Channels-last <c>[frames, height, width, H]</c>, the input's dtype (F32 or BF16).</param>
    /// <param name="input">Channels-last <c>[frames, height, width, 2H]</c>, F32 or BF16: value channels first, then gate.</param>
    /// <param name="taps">F32 taps <c>[k, k, 2H]</c>, tap-major (a PyTorch depthwise weight <c>[2H, 1, k, k]</c> transposed).</param>
    /// <param name="bias">Per-channel bias <c>[2H]</c>, F32, or null.</param>
    void DepthwiseConvGlu(Tensor output, Tensor input, Tensor taps, Tensor? bias, int frames, int height, int width, int kernel,
        bool inputSilu) =>
        DepthwiseConvGluReference.Run(output, input, taps, bias, frames, height, width, kernel, inputSilu);
}
