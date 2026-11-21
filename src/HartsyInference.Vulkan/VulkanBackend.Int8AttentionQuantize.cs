using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// INT8 attention on Vulkan, first part: the Q and K prequantization of Int8AttentionV2Reference.Quantize without key
// centering. The rotation is the backend's Hadamard stages (Shaders/hadamard.comp.glsl); the quantization has a shader
// of its own (int8_attn_quant_rows.comp.glsl), because the linear layers' quantizer does not give the reference's scales
// bit for bit.
public sealed partial class VulkanBackend
{
    /// <summary>Rotates each row of <paramref name="rows"/> and quantizes it to signed bytes with one scale a row, as
    /// <see cref="Int8AttentionV2Reference.Quantize"/> does for a query row, and for a key row under per-key scales with
    /// no smoothing: <c>scale = max|x| / 127</c>, <c>byte = clamp(roundEven(x / scale), -127, 127)</c>.</summary>
    /// <remarks>
    /// <para>The rotation runs as the backend's Hadamard stages with the reference's two scale factors folded into one.
    /// The reference halves after its four-point stage and scales again at the end; halving is exact, so the product of
    /// the two applied once gives the same F32 value.</para>
    /// <para>The scale has the reference's floor (1 below the smallest normal float) and a row holding a NaN gets a NaN
    /// scale and zero bytes, as in the reference. Head dims the reference would pad are the caller's to pad.</para>
    /// </remarks>
    /// <param name="bytes">Output, I8 <c>[N, Dp]</c>.</param>
    /// <param name="scales">Output, F32, one value a row.</param>
    /// <param name="rows">F32 <c>[N, Dp]</c> with <c>Dp</c> 64, 128 or 256.</param>
    /// <param name="rotation">A resolved transform that fits <c>Dp</c>, not <see cref="Int8Rotation.Pinned"/>.</param>
    public void QuantizeInt8AttentionRows(Tensor bytes, Tensor scales, Tensor rows, Int8Rotation rotation)
    {
        if (rows.DType != DType.F32 || rows.Shape.Rank != 2)
            throw new ArgumentException($"The rows are F32 [N, Dp]; got {rows.DType} {rows.Shape}.", nameof(rows));
        int n = (int)rows.Shape[0], dp = (int)rows.Shape[1];
        if (dp is not (64 or 128 or 256))
            throw new ArgumentOutOfRangeException(nameof(rows), dp, "The padded head dim is 64, 128 or 256.");
        if (rotation == Int8Rotation.Pinned)
            throw new ArgumentException("Resolve the pinned rotation for the call's key count first.", nameof(rotation));
        // Checks that the transform runs at this width; the key count plays no part once the rotation is resolved.
        Int8AttentionRotation.Resolve(rotation, dp, validKeys: 1);
        if (bytes.DType != DType.I8 || bytes.Shape != rows.Shape)
            throw new ArgumentException($"The bytes are I8 {rows.Shape}; got {bytes.DType} {bytes.Shape}.", nameof(bytes));
        if (scales.DType != DType.F32 || scales.ElementCount != n)
            throw new ArgumentException($"The scales are F32, one a row ({n}); got {scales.DType} {scales.Shape}.", nameof(scales));
        if (n == 0) return;
        RequireShader("int8_attn_quant_rows");

        Tensor rotated = rows;
        try
        {
            switch (rotation)
            {
                case Int8Rotation.None:
                    break;
                case Int8Rotation.H4:
                    rotated = Relabelled(rows, (long)n * dp / 4, 4);
                    Hadamard(rotated, rotated, 0.5f);
                    break;
                case Int8Rotation.H64:
                    rotated = Relabelled(rows, n, 64);
                    Hadamard(rotated, rotated, 0.5f * 0.25f);
                    break;
                case Int8Rotation.SignedH128:
                {
                    using Tensor signs = new(new TensorShape(128), DType.F32);
                    Span<float> sign = signs.AsSpan<float>();
                    for (int channel = 0; channel < 128; channel++) sign[channel] = Int8AttentionRotation.KeepsSign(channel) ? 1f : -1f;
                    rotated = new Tensor(rows.Shape, DType.F32);
                    ((HartsyInference.Core.Backends.IBackend)this).AffineBroadcastLastDim(rotated, rows, signs, null);
                    Hadamard(rotated, rotated, 0.5f * Int8AttentionRotation.H128Scale);
                    break;
                }
                case Int8Rotation.PaddedH128:
                    rotated = Relabelled(rows, 2L * n, 128);
                    Hadamard(rotated, rotated, 0.5f * Int8AttentionRotation.H128Scale);
                    break;
                default:
                    throw new ArgumentException($"{rotation} is not a resolved rotation.", nameof(rotation));
            }

            using OpScope _op = EnterOp();
            VulkanBuffer source = GetBuffer(rotated);
            VulkanBuffer byteBuffer = _xfer.AllocateDevice((ulong)n * (ulong)dp);
            VulkanBuffer scaleBuffer = _xfer.AllocateDevice((ulong)n * sizeof(float));
            try
            {
                VulkanKernel kernel = GetKernel("int8_attn_quant_rows", storageBufferCount: 3, _default1DSpec);
                Span<byte> pc = stackalloc byte[4 * 4];
                BinaryWriteUInt(pc, 0, (uint)n);
                BinaryWriteUInt(pc, 4, (uint)dp);
                BinaryWriteFloat(pc, 8, 127f);
                Span<ulong> buffers = stackalloc ulong[] { source.Handle, byteBuffer.Handle, scaleBuffer.Handle };
                // One workgroup a row, and a grid dimension is only guaranteed 65535 of them: a long key axis goes in pieces.
                for (int done = 0; done < n; done += Int8QuantizeRowsPerDispatch)
                {
                    BinaryWriteUInt(pc, 12, (uint)done);
                    Dispatch(kernel, buffers, pc, (uint)Math.Min(n - done, Int8QuantizeRowsPerDispatch), 1, 1);
                }
                CacheOutput(bytes, byteBuffer);
                CacheOutput(scales, scaleBuffer);
            }
            finally
            {
                // FreeDevice leaves a buffer the cache has taken to the cache.
                _xfer.FreeDevice(byteBuffer);
                _xfer.FreeDevice(scaleBuffer);
            }
        }
        finally
        {
            if (!ReferenceEquals(rotated, rows)) rotated.Dispose();
        }
    }

    /// <summary>Rows one dispatch of the quantizer takes: the workgroup count every Vulkan device guarantees in a dimension.</summary>
    private const int Int8QuantizeRowsPerDispatch = 65535;

    /// <summary>A copy of <paramref name="rows"/> on the backend declared as <c>[count, width]</c>, so a transform over
    /// groups of <paramref name="width"/> channels sees them as its rows.</summary>
    private Tensor Relabelled(Tensor rows, long count, int width)
    {
        Tensor copy = new(new TensorShape(count, width), DType.F32);
        try
        {
            CopyTo(copy, rows);
        }
        catch
        {
            copy.Dispose();
            throw;
        }
        return copy;
    }
}
