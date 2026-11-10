using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// Walsh-Hadamard stages, and the block-scaled FP8 and grouped linears (Shaders/hadamard.comp.glsl, fp8_block_linear.comp.glsl).
public sealed partial class VulkanBackend
{
    /// <inheritdoc/>
    public void Hadamard(Tensor output, Tensor input, float scale)
    {
        int width = HadamardReference.Validate(output, input);
        RequireShader("hadamard");
        long count = input.ElementCount;
        if (count == 0)
        {
            return;
        }
        using OpScope _op = EnterOp();
        bool inPlace = ReferenceEquals(input, output);
        VulkanBuffer inBuf = GetBuffer(input);
        VulkanBuffer outBuf = inPlace ? inBuf : _xfer.AllocateDevice((ulong)(count * sizeof(float)));
        try
        {
            VulkanKernel kernel = GetKernel("hadamard", storageBufferCount: 2, _default1DSpec);
            if (width == 1)
            {
                DispatchHadamard(kernel, inBuf, outBuf, count, width, 0u, scale);
            }
            else
            {
                VulkanBuffer from = inBuf;
                for (uint h = 1; h < width; h <<= 1)
                {
                    DispatchHadamard(kernel, from, outBuf, count / 2, width, h, h << 1 == width ? scale : 1f);
                    from = outBuf;
                }
            }
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan Hadamard dispatch failed", ex);
            if (!inPlace) outBuf.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public void Fp8BlockLinear(Tensor output, Tensor input, Tensor weight, Tensor weightScaleInv, Tensor? bias,
        int blockRows = 128, int blockCols = 128)
    {
        (int m, int n, int k) = Fp8BlockLinearReference.Validate(output, input, weight, weightScaleInv, bias, blockRows, blockCols);
        RequireShader("fp8_block_linear");
        if (m == 0 || n == 0)
        {
            return;
        }
        using OpScope _op = EnterOp();
        DispatchBlockLinear(output, input, weight, weightScaleInv, bias, m, n, k, k, n, blockRows, blockCols, "Fp8BlockLinear");
    }

    /// <inheritdoc/>
    public void BlockDiagonalLinear(Tensor output, Tensor input, Tensor weight, Tensor? weightScaleInv, int groups,
        int blockRows = 128, int blockCols = 128)
    {
        (int m, int r, int d) = BlockDiagonalLinearReference.Validate(output, input, weight, weightScaleInv, groups, blockRows, blockCols);
        RequireShader("fp8_block_linear");
        if (m == 0)
        {
            return;
        }
        using OpScope _op = EnterOp();
        DispatchBlockLinear(output, input, weight, weightScaleInv, null, m, groups * r, d, groups * d, r, blockRows, blockCols, "BlockDiagonalLinear");
    }

    // fp8_block_linear over an FP8 weight with its scales, or an F32 weight when scaleInv is null. The workgroup count in X is
    // capped at 65535 on the minimum Vulkan limit, so N is dispatched in chunks.
    private void DispatchBlockLinear(Tensor output, Tensor input, Tensor weight, Tensor? scaleInv, Tensor? bias, int m, int n, int k,
        int inWidth, int groupRows, int blockRows, int blockCols, string op)
    {
        const uint maxGroups = 65535;
        List<VulkanBuffer> temps = new();
        VulkanBuffer outBuf = _xfer.AllocateDevice((ulong)(output.ElementCount * sizeof(float)));
        try
        {
            VulkanBuffer xBuf = GetBuffer(input);
            VulkanBuffer wBuf = scaleInv is null ? GetBuffer(weight) : PaddedInput(weight, temps);
            VulkanBuffer sBuf = scaleInv is null ? Placeholder(temps) : GetBuffer(scaleInv);
            VulkanBuffer bBuf = bias is null ? Placeholder(temps) : GetBuffer(bias);
            VulkanKernel kernel = GetKernel("fp8_block_linear", storageBufferCount: 5, ReadOnlySpan<SpecConstant>.Empty);
            Span<ulong> bufs = stackalloc ulong[] { xBuf.Handle, wBuf.Handle, sBuf.Handle, bBuf.Handle, outBuf.Handle };
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            for (uint nBase = 0; nBase < n; nBase += maxGroups)
            {
                Authoring.PushConstants pc = new(pcBytes);
                pc.U32((uint)m);
                pc.U32((uint)n);
                pc.U32((uint)k);
                pc.U32((uint)blockRows);
                pc.U32((uint)blockCols);
                pc.U32((uint)((k + blockCols - 1) / blockCols));
                pc.U32(bias is null ? 0u : 1u);
                pc.U32(nBase);
                pc.U32((uint)inWidth);
                pc.U32((uint)groupRows);
                pc.U32(scaleInv is null ? 0u : 1u);
                uint groupsN = (uint)Math.Min(n - nBase, maxGroups);
                Dispatch(kernel, bufs, pc.Written, groupsN, (uint)Math.Min(m, 65535), 1);
            }
            CacheOutput(output, outBuf);
        }
        catch (Exception ex)
        {
            Logs.Error($"Vulkan {op} dispatch failed", ex);
            outBuf.Dispose();
            throw;
        }
        finally
        {
            FreeTemps(temps);
        }
    }

    // One butterfly stage (or the width-1 scale when h == 0) over `total` pairs, chunked to the dispatch limit.
    private void DispatchHadamard(VulkanKernel kernel, VulkanBuffer src, VulkanBuffer dst, long total, int width, uint h, float scale)
    {
        Span<ulong> bufs = stackalloc ulong[] { src.Handle, dst.Handle };
        Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
        for (long done = 0; done < total; done += MaxElementsPerDispatch)
        {
            Authoring.PushConstants pc = new(pcBytes);
            pc.U32((uint)total);
            pc.U32((uint)width);
            pc.U32(h);
            pc.F32(scale);
            pc.U32((uint)done);
            Dispatch(kernel, bufs, pc.Written, GroupCount(Math.Min(total - done, MaxElementsPerDispatch), LocalX1D));
        }
    }

    // The .spv is built from Shaders/ with Shaders/build.sh; until it exists the op refuses rather than failing in the loader.
    private void RequireShader(string name)
    {
        if (!File.Exists(Path.Combine(_spvDir, name + ".spv")))
            throw new NotSupportedException($"Vulkan {name} needs {name}.spv in the Spirv folder; build it with Shaders/build.sh.");
    }
}
