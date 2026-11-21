using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// Test probes for the exact operations shaders share: fmaRn (Shaders/fma_rn.glsl) and exp2Fixed
// (Shaders/exp2_fixed.glsl). Each returns the shared routine beside the device's own, so a test can hold the first to
// its host twin and count how far the second is from it.
public sealed partial class VulkanBackend
{
    /// <summary>Test probe: <c>fmaRn</c> of Shaders/fma_rn.glsl and the device's own <c>fma()</c> of each triple.</summary>
    /// <param name="emulated">Output, F32: <c>fmaRn(a, b, c)</c>.</param>
    /// <param name="native">Output, F32: the device's <c>fma(a, b, c)</c>, which it need not fuse.</param>
    /// <param name="a">F32 multiplicands.</param>
    /// <param name="b">F32 multipliers, as many.</param>
    /// <param name="c">F32 addends, as many.</param>
    public void FusedMultiplyAddProbe(Tensor emulated, Tensor native, Tensor a, Tensor b, Tensor c)
    {
        long count = a.ElementCount;
        foreach (Tensor tensor in (Tensor[])[emulated, native, a, b, c])
            if (tensor.DType != DType.F32 || tensor.ElementCount != count)
                throw new ArgumentException($"Every operand is F32 with {count} elements; got {tensor.DType} {tensor.Shape}.");
        ArgumentOutOfRangeException.ThrowIfGreaterThan(count, MaxElementsPerDispatch);
        if (count == 0) return;
        RequireShader("fma_rn_probe");

        using OpScope _op = EnterOp();
        VulkanBuffer aBuffer = GetBuffer(a), bBuffer = GetBuffer(b), cBuffer = GetBuffer(c);
        VulkanBuffer emulatedBuffer = _xfer.AllocateDevice((ulong)count * sizeof(float));
        VulkanBuffer nativeBuffer = _xfer.AllocateDevice((ulong)count * sizeof(float));
        try
        {
            VulkanKernel kernel = GetKernel("fma_rn_probe", storageBufferCount: 5, _default1DSpec);
            Span<byte> pc = stackalloc byte[4];
            BinaryWriteUInt(pc, 0, (uint)count);
            Span<ulong> buffers = stackalloc ulong[]
                { aBuffer.Handle, bBuffer.Handle, cBuffer.Handle, emulatedBuffer.Handle, nativeBuffer.Handle };
            Dispatch(kernel, buffers, pc, GroupCount(count, LocalX1D));
            CacheOutput(emulated, emulatedBuffer);
            CacheOutput(native, nativeBuffer);
        }
        finally
        {
            _xfer.FreeDevice(emulatedBuffer);
            _xfer.FreeDevice(nativeBuffer);
        }
    }

    /// <summary>Test probe: <c>exp2Fixed</c> of Shaders/exp2_fixed.glsl, whose host twin is
    /// <see cref="Int8AttentionExp2.Evaluate"/>, and the device's own <c>exp2()</c> of each argument.</summary>
    /// <param name="fixedResult">Output, F32: <c>exp2Fixed(x)</c>.</param>
    /// <param name="native">Output, F32: the device's <c>exp2(x)</c>, a driver's approximation.</param>
    /// <param name="x">F32 arguments.</param>
    public void Exp2FixedProbe(Tensor fixedResult, Tensor native, Tensor x)
    {
        long count = x.ElementCount;
        foreach (Tensor tensor in (Tensor[])[fixedResult, native, x])
            if (tensor.DType != DType.F32 || tensor.ElementCount != count)
                throw new ArgumentException($"Every operand is F32 with {count} elements; got {tensor.DType} {tensor.Shape}.");
        ArgumentOutOfRangeException.ThrowIfGreaterThan(count, MaxElementsPerDispatch);
        if (count == 0) return;
        RequireShader("exp2_fixed_probe");

        using OpScope _op = EnterOp();
        VulkanBuffer source = GetBuffer(x);
        VulkanBuffer fixedBuffer = _xfer.AllocateDevice((ulong)count * sizeof(float));
        VulkanBuffer nativeBuffer = _xfer.AllocateDevice((ulong)count * sizeof(float));
        try
        {
            VulkanKernel kernel = GetKernel("exp2_fixed_probe", storageBufferCount: 3, _default1DSpec);
            Span<byte> pc = stackalloc byte[4];
            BinaryWriteUInt(pc, 0, (uint)count);
            Span<ulong> buffers = stackalloc ulong[] { source.Handle, fixedBuffer.Handle, nativeBuffer.Handle };
            Dispatch(kernel, buffers, pc, GroupCount(count, LocalX1D));
            CacheOutput(fixedResult, fixedBuffer);
            CacheOutput(native, nativeBuffer);
        }
        finally
        {
            _xfer.FreeDevice(fixedBuffer);
            _xfer.FreeDevice(nativeBuffer);
        }
    }
}
