using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// INT8 attention on Vulkan, third part: the Q side of Int8AttentionV2Reference.Quantize. The query bytes and scales are
// the row quantizer of the first part; what is new is the dot of each query with its key mean
// (Shaders/int8_attn_query_mean_dot), taken as the reference takes it through the exact fused multiply-add of
// Shaders/fma_rn.glsl.
public sealed partial class VulkanBackend
{
    /// <summary>The dot of each query row with the key mean of its KV head, as a chain of fused multiply-adds over the
    /// channels in order: the reference's <c>QueryMeanDots</c> for one key segment.</summary>
    /// <param name="dots">Output, F32, one value a query row: <c>[heads, rows]</c>.</param>
    /// <param name="queries">F32 <c>[heads, rows, Dp]</c>, not rotated.</param>
    /// <param name="means">F32 <c>[kvHeads, Dp]</c>; <c>heads</c> is a whole multiple of <c>kvHeads</c>, and query head
    /// <c>h</c> reads the mean of KV head <c>h / (heads / kvHeads)</c>.</param>
    public void Int8AttentionQueryMeanDots(Tensor dots, Tensor queries, Tensor means)
    {
        (int heads, int rows, int dp) = Int8HeadMajor(queries, nameof(queries));
        if (means.DType != DType.F32 || means.Shape.Rank != 2 || means.Shape[1] != dp)
            throw new ArgumentException($"The means are F32 [kvHeads, {dp}]; got {means.DType} {means.Shape}.", nameof(means));
        int kvHeads = (int)means.Shape[0];
        if (heads > 0 && (kvHeads == 0 || heads % kvHeads != 0))
            throw new ArgumentException($"{heads} query heads are not a whole multiple of {kvHeads} KV heads.", nameof(means));
        long total = (long)heads * rows;
        if (dots.DType != DType.F32 || dots.ElementCount != total)
            throw new ArgumentException($"The dots are F32 [{heads}, {rows}]; got {dots.DType} {dots.Shape}.", nameof(dots));
        if (total == 0) return;
        RequireShader("int8_attn_query_mean_dot");

        using OpScope _op = EnterOp();
        VulkanBuffer source = GetBuffer(queries), meanBuffer = GetBuffer(means);
        VulkanBuffer dotBuffer = _xfer.AllocateDevice((ulong)total * sizeof(float));
        try
        {
            VulkanKernel kernel = GetKernel("int8_attn_query_mean_dot", storageBufferCount: 3, _default1DSpec);
            Span<byte> pc = stackalloc byte[5 * 4];
            BinaryWriteUInt(pc, 0, (uint)total);
            BinaryWriteUInt(pc, 4, (uint)rows);
            BinaryWriteUInt(pc, 8, (uint)dp);
            BinaryWriteUInt(pc, 12, (uint)(heads / kvHeads));
            Span<ulong> buffers = stackalloc ulong[] { source.Handle, meanBuffer.Handle, dotBuffer.Handle };
            for (long done = 0; done < total; done += MaxElementsPerDispatch)
            {
                BinaryWriteUInt(pc, 16, (uint)done);
                Dispatch(kernel, buffers, pc, GroupCount(Math.Min(total - done, MaxElementsPerDispatch), LocalX1D));
            }
            CacheOutput(dots, dotBuffer);
        }
        finally
        {
            _xfer.FreeDevice(dotBuffer);
        }
    }

    /// <summary>The Q side of <see cref="Int8AttentionV2Reference.Quantize"/> for one key segment: the dot of each query
    /// with its key mean, taken before the rotation, then each row rotated and quantized as
    /// <see cref="QuantizeInt8AttentionRows"/> does.</summary>
    /// <param name="bytes">Output, I8 <c>[heads · rows, Dp]</c>.</param>
    /// <param name="scales">Output, F32, one value a query row.</param>
    /// <param name="dots">Output, F32, one value a query row; null with <paramref name="means"/> null.</param>
    /// <param name="queries">F32 <c>[heads, rows, Dp]</c> with <c>Dp</c> 64, 128 or 256.</param>
    /// <param name="means">The key means <c>[kvHeads, Dp]</c> the keys were centered by; null where they were not.</param>
    /// <param name="rotation">A resolved transform that fits <c>Dp</c>, the one the keys took.</param>
    public void QuantizeInt8AttentionQueries(Tensor bytes, Tensor scales, Tensor? dots, Tensor queries, Tensor? means,
        Int8Rotation rotation)
    {
        (int heads, int rows, int dp) = Int8HeadMajor(queries, nameof(queries));
        if ((means is null) != (dots is null))
            throw new ArgumentException("The dots and the means are given together or not at all.", nameof(dots));
        if (means is not null) Int8AttentionQueryMeanDots(dots!, queries, means);
        if ((long)heads * rows == 0) return;
        using Tensor flat = Relabelled(queries, (long)heads * rows, dp);
        QuantizeInt8AttentionRows(bytes, scales, flat, rotation);
    }

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
