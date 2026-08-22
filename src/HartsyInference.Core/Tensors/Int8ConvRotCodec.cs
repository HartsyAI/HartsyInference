using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace HartsyInference.Core.Tensors;

/// <summary>ComfyUI's <c>int8_tensorwise</c> weight format, with and without the <c>convrot</c> Hadamard rotation (comfy-kitchen <c>tensor/int8_utils.py</c> + <c>backends/eager/quantization.py</c>).</summary>
/// <remarks><para>The quantizer stores <c>W @ Hᵀ</c> row-quantized to int8 with a per-output-row scale, and expects the
/// consumer to rotate the activation by <c>H</c> per group at runtime: <c>(x·H)·(W·Hᵀ)ᵀ = x·H·H·Wᵀ = x·Wᵀ</c>, because
/// the normalized regular Hadamard is both symmetric and orthogonal. This type provides the un-rotating dequant for
/// backends and layers that cannot take the packed path, and the matrix itself for those that can.</para>
/// <para>Rotation runs as a radix-4 butterfly, not a matrix multiply. <c>H</c> is <c>kron(h4, …, h4)</c>, so applying
/// <c>h4</c> once along each of the <c>log₄(G)</c> digit axes is the same transform at <c>O(G·log₄G)</c> instead of
/// <c>O(G²)</c> — the difference between a 4 GFLOP and a 0.07 GFLOP pass over one 4096×4096 LTX 2.5 layer.</para></remarks>
public static unsafe class Int8ConvRotCodec
{
    private static readonly ConcurrentDictionary<int, float[]> _hadamardCache = new();

    /// <summary>The normalized regular Hadamard <c>H[size, size]</c>, row-major, cached per size.</summary>
    /// <param name="size">Must be a power of four — comfy-kitchen builds <c>H</c> by Kronecker powers of a 4×4 seed.</param>
    public static float[] BuildHadamard(int size)
    {
        ValidateGroupSize(size);
        return _hadamardCache.GetOrAdd(size, static s =>
        {
            float[] h = new float[(long)s * s];
            for (int row = 0; row < s; row++)
            {
                h[(long)row * s + row] = 1.0f;
            }
            fixed (float* p = h)
            {
                for (int row = 0; row < s; row++)
                {
                    ApplyRotation(p + (long)row * s, s, s);
                }
            }
            return h;
        });
    }

    /// <summary>Whether <paramref name="size"/> is a usable ConvRot group size (a power of four, at least 4).</summary>
    public static bool IsValidGroupSize(int size) => size >= 4 && (size & (size - 1)) == 0 && (size & 0x55555554) != 0;

    /// <summary>Rotates every contiguous <paramref name="groupSize"/>-wide group of <paramref name="values"/> by <c>H</c> in place.</summary>
    public static void ApplyRotationInPlace(Span<float> values, int groupSize)
    {
        ValidateGroupSize(groupSize);
        if (values.Length % groupSize != 0)
            throw new ArgumentException($"Length {values.Length} is not a multiple of ConvRot group size {groupSize}.", nameof(values));
        fixed (float* p = values)
        {
            ApplyRotation(p, values.Length, groupSize);
        }
    }

    /// <summary>Dequantizes an int8 weight <c>[out, in]</c> and un-rotates it, producing the BF16 <c>[out, in]</c> <c>IBackend.Linear</c> consumes.</summary>
    /// <param name="rowScale">F32, one scale per output row; a single element is broadcast as a per-tensor scale.</param>
    /// <param name="convRotGroupSize">0 for a weight quantized without rotation.</param>
    public static Tensor DequantToBf16(Tensor weight, Tensor rowScale, int convRotGroupSize)
    {
        ArgumentNullException.ThrowIfNull(weight);
        ArgumentNullException.ThrowIfNull(rowScale);
        if (weight.DType != DType.I8 || weight.Shape.Rank != 2)
            throw new ArgumentException($"ConvRot weight must be I8 rank-2; got {weight.DType} {weight.Shape}.", nameof(weight));
        if (rowScale.DType != DType.F32)
            throw new ArgumentException($"ConvRot weight scale must be F32; got {rowScale.DType}.", nameof(rowScale));

        long outFeatures = weight.Shape[0];
        long inFeatures = weight.Shape[1];
        if (rowScale.ElementCount != outFeatures && rowScale.ElementCount != 1)
        {
            throw new ArgumentException(
                $"ConvRot weight scale must hold {outFeatures} row scales or a single per-tensor scale; got {rowScale.ElementCount}.",
                nameof(rowScale));
        }
        if (convRotGroupSize > 0)
        {
            ValidateGroupSize(convRotGroupSize);
            if (inFeatures % convRotGroupSize != 0)
                throw new ArgumentException(
                    $"ConvRot group size {convRotGroupSize} does not divide in_features {inFeatures}.", nameof(convRotGroupSize));
        }

        Tensor result = new Tensor(new TensorShape(outFeatures, inFeatures), DType.BF16);
        sbyte* source = (sbyte*)weight.DataPointer;
        float* scales = (float*)rowScale.DataPointer;
        bool perTensor = rowScale.ElementCount == 1;
        ushort* destination = (ushort*)result.DataPointer;

        Parallel.For(0, (int)outFeatures, () => new float[inFeatures], (row, _, scratch) =>
        {
            sbyte* sourceRow = source + row * inFeatures;
            float scale = perTensor ? scales[0] : scales[row];
            for (long column = 0; column < inFeatures; column++)
            {
                scratch[column] = sourceRow[column] * scale;
            }
            if (convRotGroupSize > 0)
            {
                fixed (float* p = scratch)
                {
                    ApplyRotation(p, (int)inFeatures, convRotGroupSize);
                }
            }
            ushort* destinationRow = destination + row * inFeatures;
            for (long column = 0; column < inFeatures; column++)
            {
                destinationRow[column] = ToBf16(scratch[column]);
            }
            return scratch;
        }, static _ => { });

        return result;
    }

    /// <summary>Radix-4 butterfly over each contiguous group, equivalent to <c>v @ H</c>.</summary>
    /// <remarks>Each stage applies the 4×4 seed along one digit axis. With
    /// <c>h4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]</c> every output is the group sum minus twice the
    /// mirrored input, and the 1/2 per stage accumulates to the 1/√G normalization.</remarks>
    private static void ApplyRotation(float* values, int count, int groupSize)
    {
        for (int groupStart = 0; groupStart < count; groupStart += groupSize)
        {
            float* group = values + groupStart;
            for (int stride = 1; stride < groupSize; stride *= 4)
            {
                int span = stride * 4;
                for (int blockStart = 0; blockStart < groupSize; blockStart += span)
                {
                    for (int lane = 0; lane < stride; lane++)
                    {
                        float* a = group + blockStart + lane;
                        float v0 = a[0], v1 = a[stride], v2 = a[2 * stride], v3 = a[3 * stride];
                        float sum = v0 + v1 + v2 + v3;
                        a[0] = 0.5f * (sum - 2.0f * v3);
                        a[stride] = 0.5f * (sum - 2.0f * v2);
                        a[2 * stride] = 0.5f * (sum - 2.0f * v1);
                        a[3 * stride] = 0.5f * (sum - 2.0f * v0);
                    }
                }
            }
        }
    }

    private static void ValidateGroupSize(int size)
    {
        if (!IsValidGroupSize(size))
            throw new ArgumentOutOfRangeException(nameof(size), size, "ConvRot group size must be a power of four (4, 16, 64, 256, …).");
    }

    /// <summary>Truncating F32→BF16, matching <see cref="Tensor.CastTo"/> bit for bit.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ushort ToBf16(float value) => (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16);
}
