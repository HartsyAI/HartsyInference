using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>The BF16-output depthwise Conv2D on the host. Operands are rounded to BF16 as the device cast does, accumulated in F32
/// in the F32 reference's tap order, and rounded once on output.</summary>
internal static unsafe class DepthwiseConv2dReference
{
    /// <summary>Computes the BF16 <paramref name="output"/> from F32 or BF16 operands.</summary>
    /// <exception cref="NotSupportedException">An operand is neither F32 nor BF16.</exception>
    internal static void ComputeBf16(Tensor output, Tensor input, Tensor weight, Tensor? bias, int strideH, int strideW, int padH, int padW)
    {
        if (!IsHostOperand(input.DType) || !IsHostOperand(weight.DType) || (bias is not null && !IsHostOperand(bias.DType)))
            throw new NotSupportedException(
                $"Conv2dDepthwise BF16 output takes F32 or BF16 operands — got input={input.DType.Name}, weight={weight.DType.Name}, bias={bias?.DType.Name ?? "none"}.");

        DepthwiseConv2dGeometry g = DepthwiseConv2dContract.Validate(input, weight, bias, output);
        void* src = input.DataPointer;
        void* kernels = weight.DataPointer;
        void* biases = bias is null ? null : bias.DataPointer;
        DType biasDtype = bias is null ? DType.F32 : bias.DType;
        ushort* dst = (ushort*)output.DataPointer;

        for (int b = 0; b < g.Batch; b++)
        for (int ch = 0; ch < g.Channels; ch++)
        {
            long srcPlane = ((long)b * g.Channels + ch) * g.InH * g.InW;
            long dstPlane = ((long)b * g.Channels + ch) * g.OutH * g.OutW;
            long kernel = (long)ch * g.KernelH * g.KernelW;
            float biasValue = biases is null ? 0f : Operand(biases, biasDtype, ch);

            for (int oy = 0; oy < g.OutH; oy++)
            {
                int iy0 = oy * strideH - padH;
                for (int ox = 0; ox < g.OutW; ox++)
                {
                    int ix0 = ox * strideW - padW;
                    float v = biasValue;
                    for (int ky = 0; ky < g.KernelH; ky++)
                    {
                        int iy = iy0 + ky;
                        if (iy < 0 || iy >= g.InH) continue;
                        for (int kx = 0; kx < g.KernelW; kx++)
                        {
                            int ix = ix0 + kx;
                            if (ix < 0 || ix >= g.InW) continue;
                            v += Operand(kernels, weight.DType, kernel + ky * g.KernelW + kx)
                                * Operand(src, input.DType, srcPlane + (long)iy * g.InW + ix);
                        }
                    }
                    dst[dstPlane + oy * g.OutW + ox] = BFloat16Bits.FromFloat(v);
                }
            }
        }
    }

    private static bool IsHostOperand(DType dtype) => dtype == DType.F32 || dtype == DType.BF16;

    /// <summary>Element <paramref name="index"/> as the device cast sees it: rounded to BF16.</summary>
    private static float Operand(void* data, DType dtype, long index)
    {
        float value = dtype == DType.BF16 ? BFloat16Bits.ToFloat(((ushort*)data)[index]) : ((float*)data)[index];
        return BFloat16Bits.ToFloat(BFloat16Bits.FromFloat(value));
    }
}
