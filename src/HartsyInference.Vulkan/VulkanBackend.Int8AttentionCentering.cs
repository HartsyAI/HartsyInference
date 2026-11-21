using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// INT8 attention on Vulkan, second part: the key centering of Int8AttentionV2Reference.Quantize for one key segment, and
// its value quantization for u8 probabilities (Shaders/int8_attn_key_mean, int8_attn_value_scales,
// int8_attn_quant_values). With the row quantizer of the first part this gives the reference's K and V bytes.
public sealed partial class VulkanBackend
{
    /// <summary>The channel mean of each head's keys, in the reference's summation order: partial sums over runs of 64 keys,
    /// added in order, over the key count.</summary>
    /// <param name="means">Output, F32 <c>[heads, Dp]</c>.</param>
    /// <param name="keys">F32 <c>[heads, keys, Dp]</c>.</param>
    public void Int8AttentionKeyMeans(Tensor means, Tensor keys)
    {
        (int heads, int count, int dp) = Int8HeadMajor(keys, nameof(keys));
        if (means.DType != DType.F32 || means.ElementCount != (long)heads * dp)
            throw new ArgumentException($"The means are F32 [{heads}, {dp}]; got {means.DType} {means.Shape}.", nameof(means));
        if (heads == 0) return;
        RequireShader("int8_attn_key_mean");

        using OpScope _op = EnterOp();
        VulkanBuffer source = GetBuffer(keys);
        VulkanBuffer meanBuffer = _xfer.AllocateDevice((ulong)heads * (ulong)dp * sizeof(float));
        try
        {
            VulkanKernel kernel = GetKernel("int8_attn_key_mean", storageBufferCount: 2, _default1DSpec);
            Span<byte> pc = stackalloc byte[3 * 4];
            BinaryWriteUInt(pc, 0, (uint)heads);
            BinaryWriteUInt(pc, 4, (uint)count);
            BinaryWriteUInt(pc, 8, (uint)dp);
            Span<ulong> buffers = stackalloc ulong[] { source.Handle, meanBuffer.Handle };
            Dispatch(kernel, buffers, pc, GroupCount((long)heads * dp, LocalX1D));
            CacheOutput(means, meanBuffer);
        }
        finally
        {
            _xfer.FreeDevice(meanBuffer);
        }
    }

    /// <summary>The K side of <see cref="Int8AttentionV2Reference.Quantize"/> with keys smoothed by their mean over one
    /// segment and per-key scales: the mean of each head's keys is subtracted, each row is rotated, and it is quantized
    /// as <see cref="QuantizeInt8AttentionRows"/> does.</summary>
    /// <param name="bytes">Output, I8 <c>[heads · keys, Dp]</c>.</param>
    /// <param name="scales">Output, F32, one value a key.</param>
    /// <param name="means">Output, F32 <c>[heads, Dp]</c>: what was subtracted, which the attention needs again.</param>
    /// <param name="keys">F32 <c>[heads, keys, Dp]</c> with <c>Dp</c> 64, 128 or 256.</param>
    /// <param name="rotation">A resolved transform that fits <c>Dp</c>.</param>
    public void QuantizeInt8AttentionKeys(Tensor bytes, Tensor scales, Tensor means, Tensor keys, Int8Rotation rotation)
    {
        (int heads, int count, int dp) = Int8HeadMajor(keys, nameof(keys));
        Int8AttentionKeyMeans(means, keys);
        if ((long)heads * count == 0) return;

        using Tensor negated = new(means.Shape, DType.F32);
        Scale(negated, means, -1f);
        // key - mean, one mean a head: an add of the negated mean broadcast over the head's rows. Both steps are exact.
        using Tensor centered = new(keys.Shape, DType.F32);
        NormModulate(NormModulateCall.Rows(centered, keys, NormKind.None, 0f)
            .WithModulation(ModulationSpec.Broadcast(keys, null, negated, ScaleMode.Raw)));
        using Tensor rows = Relabelled(centered, (long)heads * count, dp);
        QuantizeInt8AttentionRows(bytes, scales, rows, rotation);
    }

    /// <summary>The V side of <see cref="Int8AttentionV2Reference.Quantize"/> for u8 probabilities: one scale a channel for
    /// each block of <paramref name="blockKeys"/> keys of a head (<c>max|v| / 127</c>, with the reference's floor and its
    /// NaN rule) and signed bytes. V is never rotated.</summary>
    /// <param name="bytes">Output, I8, as many elements as <paramref name="values"/>.</param>
    /// <param name="scales">Output, F32 <c>[heads, blocks, Dp]</c>, <c>blocks</c> the block count that covers the keys.</param>
    /// <param name="values">F32 <c>[heads, keys, Dp]</c> with <c>Dp</c> 64, 128 or 256.</param>
    /// <param name="blockKeys">Keys a block; the key count itself gives one scale a channel for the whole head.</param>
    public void QuantizeInt8AttentionValues(Tensor bytes, Tensor scales, Tensor values, int blockKeys)
    {
        (int heads, int count, int dp) = Int8HeadMajor(values, nameof(values));
        ArgumentOutOfRangeException.ThrowIfLessThan(blockKeys, 1);
        int blocks = Math.Max(1, (count + blockKeys - 1) / blockKeys);
        long elements = (long)heads * count * dp, scaleCount = (long)heads * blocks * dp;
        if (bytes.DType != DType.I8 || bytes.ElementCount != elements)
            throw new ArgumentException($"The bytes are I8 with {elements} elements; got {bytes.DType} {bytes.Shape}.", nameof(bytes));
        if (scales.DType != DType.F32 || scales.ElementCount != scaleCount)
            throw new ArgumentException($"The scales are F32 [{heads}, {blocks}, {dp}]; got {scales.DType} {scales.Shape}.", nameof(scales));
        if (elements == 0) return;
        RequireShader("int8_attn_value_scales");
        RequireShader("int8_attn_quant_values");

        using OpScope _op = EnterOp();
        VulkanBuffer source = GetBuffer(values);
        VulkanBuffer scaleBuffer = _xfer.AllocateDevice((ulong)scaleCount * sizeof(float));
        VulkanBuffer inverseBuffer = _xfer.AllocateDevice((ulong)scaleCount * sizeof(float));
        VulkanBuffer byteBuffer = _xfer.AllocateDevice((ulong)elements);
        try
        {
            VulkanKernel stats = GetKernel("int8_attn_value_scales", storageBufferCount: 3, _default1DSpec);
            Span<byte> pc = stackalloc byte[6 * 4];
            BinaryWriteUInt(pc, 0, (uint)heads);
            BinaryWriteUInt(pc, 4, (uint)count);
            BinaryWriteUInt(pc, 8, (uint)dp);
            BinaryWriteUInt(pc, 12, (uint)blockKeys);
            BinaryWriteUInt(pc, 16, (uint)blocks);
            BinaryWriteFloat(pc, 20, 127f);
            Span<ulong> statBuffers = stackalloc ulong[] { source.Handle, scaleBuffer.Handle, inverseBuffer.Handle };
            Dispatch(stats, statBuffers, pc, GroupCount(scaleCount, LocalX1D));

            VulkanKernel quantize = GetKernel("int8_attn_quant_values", storageBufferCount: 3, _default1DSpec);
            Span<ulong> quantBuffers = stackalloc ulong[] { source.Handle, inverseBuffer.Handle, byteBuffer.Handle };
            long words = elements / 4;
            for (long done = 0; done < words; done += MaxElementsPerDispatch)
            {
                BinaryWriteUInt(pc, 20, (uint)done);   // the quantize pass reads a word base where the stats pass read the range
                Dispatch(quantize, quantBuffers, pc, GroupCount(Math.Min(words - done, MaxElementsPerDispatch), LocalX1D));
            }
            CacheOutput(bytes, byteBuffer);
            CacheOutput(scales, scaleBuffer);
        }
        finally
        {
            // FreeDevice leaves a buffer the cache has taken to the cache.
            _xfer.FreeDevice(byteBuffer);
            _xfer.FreeDevice(scaleBuffer);
            _xfer.FreeDevice(inverseBuffer);
        }
    }

    /// <summary>Checks a head-major F32 <c>[heads, keys, Dp]</c> operand with a padded head dim.</summary>
    private static (int Heads, int Keys, int Dp) Int8HeadMajor(Tensor operand, string name)
    {
        if (operand.DType != DType.F32 || operand.Shape.Rank != 3)
            throw new ArgumentException($"The operand is F32 [heads, keys, Dp]; got {operand.DType} {operand.Shape}.", name);
        int dp = (int)operand.Shape[2];
        if (dp is not (64 or 128 or 256))
            throw new ArgumentOutOfRangeException(name, dp, "The padded head dim is 64, 128 or 256.");
        return ((int)operand.Shape[0], (int)operand.Shape[1], dp);
    }
}
