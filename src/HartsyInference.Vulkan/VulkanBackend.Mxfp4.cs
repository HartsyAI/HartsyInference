using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;

namespace HartsyInference.Vulkan;

// The MXFP4-packed linear (IBackend.Mxfp4LinearF32) on the device: one workgroup per output row (Shaders/mxfp4_linear.comp.glsl).
public sealed partial class VulkanBackend
{
    /// <summary>Invocations per output row's workgroup; the shader reduces across them.</summary>
    private const uint Mxfp4LinearLocal = 256;

    /// <inheritdoc/>
    public void Mxfp4LinearF32(Tensor output, Tensor input, Tensor packedWeight, QuantRecipe recipe)
    {
        using OpScope _op = EnterOp();
        Mxfp4LinearReference.Validate(output, input, packedWeight, recipe, out int n, out int rows, out int cols);
        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer packedBuf = GetBuffer(packedWeight);
        VulkanBuffer scaleBuf = GetBuffer(recipe.Scale!);
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(output.ElementCount * sizeof(float)));
        try
        {
            VulkanKernel kernel = GetKernel("mxfp4_linear", storageBufferCount: 4, RowSpec(Mxfp4LinearLocal));
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)n);
            pc.U32((uint)rows);
            pc.U32((uint)cols);
            pc.U32((uint)packedWeight.ElementCount);
            pc.U32((uint)recipe.Scale!.ElementCount);
            Span<ulong> bufs = stackalloc ulong[] { outBuf.Handle, inBuf.Handle, packedBuf.Handle, scaleBuf.Handle };
            Dispatch(kernel, bufs, pc.Written, (uint)rows);
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan Mxfp4LinearF32 dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }
}
