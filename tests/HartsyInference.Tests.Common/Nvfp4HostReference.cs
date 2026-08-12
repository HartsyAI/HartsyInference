using System.Reflection;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.BlockScale;

namespace HartsyInference.Tests.Common;

/// <summary>Host dequant of a rank-2 nvfp4 weight, for tests that gate the RESIDENT nvfp4 path (CUDA kernel and
/// <see cref="Nvfp4ResidentCodec"/>) against an implementation already validated on a real checkpoint.</summary>
/// <remarks><para><see cref="Bf16Words"/> reaches <c>Nvfp4Linear.DequantBf16</c> — the transcription verified
/// against <c>qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors</c>, and the reference every bit-exactness gate is
/// stated against. <see cref="ExactF32"/> repeats the same arithmetic un-narrowed so an F16 comparison measures
/// F16's rounding rather than BF16's; it decodes E4M3 through <see cref="Tensor.CastTo"/> and E2M1 through the
/// literal table below rather than reusing either implementation's constants.</para>
/// <para>It deliberately does NOT use <c>Nvfp4Codec.DequantExpertSlice</c>, whose E4M3 table decodes
/// <c>0x7F</c>/<c>0xFF</c> as NaN where the whole resident path (and <c>Tensor.CastTo</c>) decodes them as
/// ±480.</para></remarks>
public static unsafe class Nvfp4HostReference
{
    /// <summary>Nibble → E2M1 value; bit 3 is the sign.</summary>
    private static readonly float[] E2M1 =
    [
        +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    ];

    /// <summary>Raw BF16 words of the reference dequant of a U8 <c>[N, K/2]</c> weight.</summary>
    /// <param name="packed">U8 <c>[N, K/2]</c> — the on-disk packing, NOT the relabelled F4E2M1 view.</param>
    public static ushort[] Bf16Words(Tensor packed, Tensor blockScale, Tensor globalScale)
    {
        ArgumentNullException.ThrowIfNull(packed);
        ArgumentNullException.ThrowIfNull(blockScale);
        ArgumentNullException.ThrowIfNull(globalScale);
        MethodInfo dequant = typeof(Nvfp4Linear).GetMethod("DequantBf16", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("Nvfp4Linear.DequantBf16 is gone — the nvfp4 host reference moved.");

        long outFeatures = packed.Shape[0];
        long inFeatures = packed.Shape[1] * 2;
        using Tensor destination = new Tensor(new TensorShape(outFeatures, inFeatures), DType.BF16);
        using Tensor packed3 = packed.Reshape(new TensorShape(1, outFeatures, packed.Shape[1]));
        using Tensor scale3 = ReshapedScale(blockScale);
        dequant.Invoke(null, [packed3, scale3, globalScale, destination]);
        return destination.AsReadOnlySpan<ushort>().ToArray();
    }

    /// <summary>Un-narrowed F32 reference dequant of a U8 <c>[N, K/2]</c> weight.</summary>
    public static float[] ExactF32(Tensor packed, Tensor blockScale, Tensor globalScale)
    {
        ArgumentNullException.ThrowIfNull(packed);
        ArgumentNullException.ThrowIfNull(blockScale);
        ArgumentNullException.ThrowIfNull(globalScale);

        long outFeatures = packed.Shape[0];
        long packedCols = packed.Shape[1];
        long inFeatures = packedCols * 2;
        long paddedCols = blockScale.Shape[1];
        float[] e4m3 = DecodeAllE4M3Bytes();
        float scaleFactor = blockScale.Fp8ScaleFactor;
        float global = ((float*)globalScale.DataPointer)[0];
        byte* weightBytes = (byte*)packed.DataPointer;
        byte* scaleBytes = (byte*)blockScale.DataPointer;

        float[] result = new float[outFeatures * inFeatures];
        for (long row = 0; row < outFeatures; row++)
        {
            for (long col = 0; col < packedCols; col++)
            {
                // Left to right, matching the order the kernel and the host reference form the product in.
                float scale = e4m3[scaleBytes[BlockScaleSwizzle.SwizzledIndex(row, col / 8, paddedCols)]]
                    * scaleFactor * global;
                byte word = weightBytes[row * packedCols + col];
                result[row * inFeatures + 2 * col] = E2M1[(word >> 4) & 0x0F] * scale;
                result[row * inFeatures + 2 * col + 1] = E2M1[word & 0x0F] * scale;
            }
        }
        return result;
    }

    private static float[] DecodeAllE4M3Bytes()
    {
        using Tensor bytes = new Tensor(new TensorShape(256), DType.F8E4M3);
        byte* p = (byte*)bytes.DataPointer;
        for (int i = 0; i < 256; i++) p[i] = (byte)i;
        using Tensor decoded = bytes.CastTo(DType.F32);
        return decoded.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>Rank-3 view of a rank-2 block scale, carrying <see cref="Tensor.Fp8ScaleFactor"/> across by hand —
    /// <see cref="Tensor.Reshape"/> drops it, and it is a factor of the dequant the resident path does apply.</summary>
    private static Tensor ReshapedScale(Tensor blockScale)
    {
        Tensor view = blockScale.Reshape(new TensorShape(1, blockScale.Shape[0], blockScale.Shape[1]));
        view.Fp8ScaleFactor = blockScale.Fp8ScaleFactor;
        return view;
    }
}
