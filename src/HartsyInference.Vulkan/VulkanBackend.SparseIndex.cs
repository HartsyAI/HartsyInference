using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// Sparse-attention index building: the exact per-row top-k and the window/top-k merge (Shaders/sparse_topk.comp.glsl,
// merge_window_topk.comp.glsl).
public sealed partial class VulkanBackend
{
    /// <inheritdoc/>
    public void SparseTopK(Tensor indices, Tensor scores, Tensor limits, int topK, int offset)
    {
        (int rows, int cols, int k) = SparseTopKReference.Validate(indices, scores, limits, topK);
        RequireShader("sparse_topk");
        if (rows == 0 || k == 0)
        {
            return;
        }
        const int maxGroups = 65535;
        using OpScope _op = EnterOp();
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(indices.ElementCount * sizeof(int)));
        try
        {
            VulkanBuffer sBuf = GetBuffer(scores);
            VulkanBuffer lBuf = GetBuffer(limits);
            VulkanKernel kernel = GetKernel("sparse_topk", storageBufferCount: 3, ReadOnlySpan<SpecConstant>.Empty);
            Span<ulong> bufs = stackalloc ulong[] { sBuf.Handle, lBuf.Handle, outBuf.Handle };
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            for (int rowBase = 0; rowBase < rows; rowBase += maxGroups)
            {
                Authoring.PushConstants pc = new(pcBytes);
                pc.U32((uint)cols);
                pc.U32((uint)k);
                pc.I32(offset);
                pc.U32((uint)rowBase);
                Dispatch(kernel, bufs, pc.Written, (uint)Math.Min(rows - rowBase, maxGroups), 1, 1);
            }
            CacheOutput(indices, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan SparseTopK dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public void MergeWindowTopK(Tensor rows, Tensor topk, Tensor positions, int window)
    {
        (int tokens, int cols, int width) = WindowTopKMergeReference.Validate(rows, topk, positions, window);
        RequireShader("merge_window_topk");
        long total = (long)tokens * width;
        if (total == 0)
        {
            return;
        }
        using OpScope _op = EnterOp();
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(total * sizeof(int)));
        try
        {
            VulkanBuffer tBuf = GetBuffer(topk);
            VulkanBuffer pBuf = GetBuffer(positions);
            VulkanKernel kernel = GetKernel("merge_window_topk", storageBufferCount: 3, _default1DSpec);
            Span<ulong> bufs = stackalloc ulong[] { tBuf.Handle, pBuf.Handle, outBuf.Handle };
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            for (long done = 0; done < total; done += MaxElementsPerDispatch)
            {
                Authoring.PushConstants pc = new(pcBytes);
                pc.U32((uint)total);
                pc.U32((uint)cols);
                pc.I32(window);
                pc.U32((uint)width);
                pc.U32((uint)done);
                Dispatch(kernel, bufs, pc.Written, GroupCount(Math.Min(total - done, MaxElementsPerDispatch), LocalX1D));
            }
            CacheOutput(rows, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan MergeWindowTopK dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
    }
}
