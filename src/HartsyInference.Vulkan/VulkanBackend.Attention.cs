using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// Hyper-connection mixing and the other DeepSeek-V4.1 attention-side primitives. Validation is the Core reference's own
// so the errors match; element-parallel kernels take an elemBase so a call larger than one dispatch can carry is split.
public sealed partial class VulkanBackend
{
    /// <summary>Elements one 1D dispatch covers: the guaranteed workgroup-count limit times the workgroup width.</summary>
    private const long MaxElementsPerDispatch = MaxWorkgroupsX * LocalX1D;

    /// <inheritdoc/>
    public void HcSplitSinkhorn(Tensor pre, Tensor post, Tensor comb, Tensor mixes, Tensor scale, Tensor bias, int hc,
        int iters, float eps)
    {
        HcReference.ValidateSplit(pre, post, comb, mixes, scale, bias, hc, iters);
        using OpScope _op = EnterOp();
        long tokens = mixes.ElementCount / ((2 + hc) * hc);
        VulkanBuffer mixesBuf = GetBuffer(mixes);
        VulkanBuffer scaleBuf = GetBuffer(scale);
        VulkanBuffer biasBuf = GetBuffer(bias);
        VulkanBuffer preBuf = _xfer.AllocateDevice((ulong)(pre.ElementCount * sizeof(float)));
        VulkanBuffer postBuf = _xfer.AllocateDevice((ulong)(post.ElementCount * sizeof(float)));
        VulkanBuffer combBuf = _xfer.AllocateDevice((ulong)(comb.ElementCount * sizeof(float)));
        try
        {
            VulkanKernel k = GetKernel("hc_split_sinkhorn", storageBufferCount: 6, _default1DSpec);
            Span<ulong> bufs = stackalloc ulong[]
            {
                preBuf.Handle, postBuf.Handle, combBuf.Handle, mixesBuf.Handle, scaleBuf.Handle, biasBuf.Handle,
            };
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            for (long done = 0; done < tokens; done += MaxElementsPerDispatch)
            {
                Authoring.PushConstants pc = new(pcBytes);
                pc.U32((uint)tokens);
                pc.U32((uint)hc);
                pc.U32((uint)iters);
                pc.F32(eps);
                pc.U32((uint)done);
                Dispatch(k, bufs, pc.Written, GroupCount(Math.Min(tokens - done, MaxElementsPerDispatch), LocalX1D));
            }
            CacheOutput(pre, preBuf);
            CacheOutput(post, postBuf);
            CacheOutput(comb, combBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan HcSplitSinkhorn dispatch failed", ex);
            preBuf.Dispose();
            postBuf.Dispose();
            combBuf.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public void HcPreMix(Tensor output, Tensor x, Tensor pre)
    {
        HcReference.ValidatePreMix(output, x, pre);
        using OpScope _op = EnterOp();
        long total = output.ElementCount;
        if (total == 0)
        {
            return;
        }
        VulkanBuffer xBuf = GetBuffer(x);
        VulkanBuffer preBuf = GetBuffer(pre);
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(total * sizeof(float)));
        try
        {
            VulkanKernel k = GetKernel("hc_pre_mix", storageBufferCount: 3, _default1DSpec);
            Span<ulong> bufs = stackalloc ulong[] { outBuf.Handle, xBuf.Handle, preBuf.Handle };
            DispatchElementwiseHc(k, bufs, total, (uint)x.Shape[1], (uint)x.Shape[2]);
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan HcPreMix dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public void HcPostMix(Tensor output, Tensor x, Tensor residual, Tensor post, Tensor comb)
    {
        HcReference.ValidatePostMix(output, x, residual, post, comb);
        using OpScope _op = EnterOp();
        if (ReferenceEquals(output, residual))
        {
            throw new ArgumentException("HcPostMix output must not alias the residual.", nameof(output));
        }
        long total = output.ElementCount;
        if (total == 0)
        {
            return;
        }
        VulkanBuffer xBuf = GetBuffer(x);
        VulkanBuffer resBuf = GetBuffer(residual);
        VulkanBuffer postBuf = GetBuffer(post);
        VulkanBuffer combBuf = GetBuffer(comb);
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(total * sizeof(float)));
        try
        {
            VulkanKernel k = GetKernel("hc_post_mix", storageBufferCount: 5, _default1DSpec);
            Span<ulong> bufs = stackalloc ulong[] { outBuf.Handle, xBuf.Handle, resBuf.Handle, postBuf.Handle, combBuf.Handle };
            DispatchElementwiseHc(k, bufs, total, (uint)residual.Shape[1], (uint)residual.Shape[2]);
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan HcPostMix dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    // total / hc / dim / elemBase push layout shared by the two element-parallel hyper-connection kernels.
    private void DispatchElementwiseHc(VulkanKernel k, Span<ulong> bufs, long total, uint hc, uint dim)
    {
        Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
        for (long done = 0; done < total; done += MaxElementsPerDispatch)
        {
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)total);
            pc.U32(hc);
            pc.U32(dim);
            pc.U32((uint)done);
            Dispatch(k, bufs, pc.Written, GroupCount(Math.Min(total - done, MaxElementsPerDispatch), LocalX1D));
        }
    }
}
