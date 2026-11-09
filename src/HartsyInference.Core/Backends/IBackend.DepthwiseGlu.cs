using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Channels-last depthwise convolution fused with a gated linear unit; the host reference is the default.
public partial interface IBackend
{
    /// <summary>Depthwise <c>k×k</c> convolution (stride 1, zero padding <c>k/2</c>) over channels-last frames, then a GLU:
    /// with <c>d</c> the convolution output (bias added), <c>output = d[..., :H] · SiLU(d[..., H:])</c>. The convolution's
    /// output is never stored; each output channel reads its value and gate channels and writes their product.</summary>
    /// <param name="output">Channels-last <c>[frames, height, width, H]</c>, the input's dtype (F32 or BF16).</param>
    /// <param name="input">Channels-last <c>[frames, height, width, 2H]</c>, F32 or BF16: value channels first, then gate.</param>
    /// <param name="weight">Depthwise taps <c>[2H, 1, k, k]</c>, the input's dtype or F32.</param>
    /// <param name="bias">Per-channel bias <c>[2H]</c>, F32, or null.</param>
    void DepthwiseConvGlu(Tensor output, Tensor input, Tensor weight, Tensor? bias, int frames, int height, int width, int kernel) =>
        DepthwiseConvGluReference.Run(output, input, weight, bias, frames, height, width, kernel);
}
