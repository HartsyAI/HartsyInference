using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>Admission for the CUDA depthwise Conv2D: the output dtype selects the kernel, and the operands are cast to it.</summary>
internal static class DepthwiseConv2dPlan
{
    /// <summary>Checks the output dtype has a depthwise kernel, then the shared geometry contract.</summary>
    /// <exception cref="NotSupportedException">The output dtype is not F32, F16 or BF16.</exception>
    /// <exception cref="ArgumentException">A tensor's rank, weight layout, or channel or bias count is wrong.</exception>
    internal static DepthwiseConv2dGeometry Validate(Tensor input, Tensor weight, Tensor? bias, Tensor output)
    {
        if (!IsSupportedOutput(output.DType))
            throw new NotSupportedException($"CUDA Conv2dDepthwise supports F32/F16/BF16 output — got {output.DType}.");
        return DepthwiseConv2dContract.Validate(input, weight, bias, output);
    }

    internal static bool IsSupportedOutput(DType dtype) => dtype == DType.F32 || dtype == DType.F16 || dtype == DType.BF16;
}
