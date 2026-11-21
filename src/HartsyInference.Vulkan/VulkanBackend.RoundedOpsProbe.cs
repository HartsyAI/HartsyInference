using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// A diagnostic entry (Shaders/rounded_ops_probe.comp.glsl). No model calls it.
public sealed partial class VulkanBackend
{
    /// <summary>Evaluates, element by element, the backend's correctly rounded divide and square root and the device's own:
    /// <paramref name="quotient"/> = <c>divRn(a, b)</c>, <paramref name="root"/> = <c>sqrtRn(a)</c>,
    /// <paramref name="rawQuotient"/> = <c>a / b</c>, <paramref name="rawRoot"/> = <c>sqrt(a)</c>. The shaders whose results
    /// must equal a host reference bit for bit use the two helpers; this lets a test check them directly and measure how
    /// often the raw operations are off on the device at hand. All six tensors are F32 with one element count.</summary>
    public void RoundedOpsProbe(Tensor quotient, Tensor root, Tensor rawQuotient, Tensor rawRoot, Tensor a, Tensor b)
    {
        long count = a.ElementCount;
        foreach (Tensor tensor in (ReadOnlySpan<Tensor>)[quotient, root, rawQuotient, rawRoot, a, b])
            if (tensor.DType != DType.F32 || tensor.ElementCount != count)
                throw new ArgumentException($"The probe takes six F32 tensors of one size; got {tensor.DType} {tensor.Shape} beside {count} elements.");
        if (count == 0) return;
        if (count > MaxElementsPerDispatch)
            throw new ArgumentException($"The probe runs one dispatch of at most {MaxElementsPerDispatch} elements; got {count}.", nameof(a));
        RequireShader("rounded_ops_probe");

        using OpScope _op = EnterOp();
        VulkanBuffer aBuf = GetBuffer(a), bBuf = GetBuffer(b);
        ulong bytes = (ulong)count * sizeof(float);
        VulkanBuffer qBuf = _xfer.AllocateDevice(bytes), rBuf = _xfer.AllocateDevice(bytes);
        VulkanBuffer rawQBuf = _xfer.AllocateDevice(bytes), rawRBuf = _xfer.AllocateDevice(bytes);
        try
        {
            VulkanKernel kernel = GetKernel("rounded_ops_probe", storageBufferCount: 6, _default1DSpec);
            Span<byte> pc = stackalloc byte[4];
            BinaryWriteUInt(pc, 0, (uint)count);
            Span<ulong> buffers = stackalloc ulong[] { aBuf.Handle, bBuf.Handle, qBuf.Handle, rBuf.Handle, rawQBuf.Handle, rawRBuf.Handle };
            Dispatch(kernel, buffers, pc, GroupCount(count, LocalX1D));
            CacheOutput(quotient, qBuf);
            CacheOutput(root, rBuf);
            CacheOutput(rawQuotient, rawQBuf);
            CacheOutput(rawRoot, rawRBuf);
        }
        finally
        {
            // FreeDevice leaves a buffer the cache has taken to the cache.
            _xfer.FreeDevice(qBuf);
            _xfer.FreeDevice(rBuf);
            _xfer.FreeDevice(rawQBuf);
            _xfer.FreeDevice(rawRBuf);
        }
    }
}
