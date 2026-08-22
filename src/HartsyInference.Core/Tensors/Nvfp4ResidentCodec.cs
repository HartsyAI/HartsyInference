using System.Runtime.CompilerServices;

namespace HartsyInference.Core.Tensors;

/// <summary>ComfyUI's <c>nvfp4</c> weight format read back on the HOST, for backends and layers that cannot consume a <see cref="DType.F4E2M1"/> weight packed.</summary>
/// <remarks><para>Sibling of <see cref="Int8ConvRotCodec"/> and here for the same reason: a backend needs a dequant
/// fallback for the layers its packed path refuses, and the backend packages reference only Core. The eager
/// load-time dequant used by the checkpoint converters lives in <c>HartsyInference.ModelAssets.Nvfp4.Nvfp4Codec</c>
/// (which additionally handles the rank-3 MoE expert bank layout); this type covers only the rank-2 Linear case.</para>
/// <para><b>Format.</b> The weight is U8 <c>[N, K/2]</c> on disk, two E2M1 nibbles per byte with the <b>high nibble
/// holding the even element</b>, relabelled to <see cref="DType.F4E2M1"/> <c>[N, K]</c> at load (see
/// <see cref="Tensor.ReinterpretAs"/>). Companions ride on <see cref="Tensor.QuantInfo"/>: an F8-E4M3
/// <c>weight_scale</c> holding one scale per 16 input elements in NVIDIA's swizzled blocked layout, and an F32
/// scalar <c>weight_scale_2</c>. Effective value =
/// <c>e2m1(nibble) · e4m3(blockScale[swizzled]) · blockScale.Fp8ScaleFactor · globalScale</c>.</para>
/// <para><b>Duplicated swizzle.</b> <see cref="SwizzledScaleIndex"/> is the same permutation as
/// <c>ModelAssets.BlockScale.BlockScaleSwizzle.SwizzledIndex</c>. It is restated here because Core cannot reference
/// ModelAssets and the dependency runs the other way; the single-home fix is to make that type forward to this
/// one.</para></remarks>
public static unsafe class Nvfp4ResidentCodec
{
    /// <summary>Input elements covered by one E4M3 block scale.</summary>
    public const int GroupSize = 16;

    /// <summary>Nibble → E2M1 value; bit 3 is the sign, so entries 8-15 mirror 0-7 (including the signed zero).</summary>
    public static readonly float[] E2M1Lut =
    [
        +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    ];

    /// <summary>E4M3 byte → float, taken from <see cref="Tensor.CastTo"/> so every consumer decodes identically.</summary>
    /// <remarks>E4M3FN as this repo defines it everywhere: no Inf, and <c>0x7F</c> is the maximum magnitude 480
    /// rather than a NaN encoding (<c>Tensor</c>'s float→E4M3 direction saturates NaN to that same byte).</remarks>
    public static readonly float[] E4M3Decode = BuildE4M3Decode();

    /// <summary>Flat element index of the scale for a logical <c>(row, blockColumn)</c> inside the swizzled <c>[paddedRows, paddedCols]</c> scale buffer.</summary>
    /// <param name="paddedCols">The scale tensor's stored last-dim length (always a multiple of 4).</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long SwizzledScaleIndex(long row, long blockColumn, long paddedCols)
    {
        long ncb = paddedCols / 4;
        long rb = row / 128;
        long r128 = row % 128;
        long a = r128 / 32;
        long b = r128 % 32;
        long cb = blockColumn / 4;
        long d = blockColumn % 4;
        long g = rb * ncb + cb;
        return (g * 32 + b) * 16 + a * 4 + d;
    }

    /// <summary>Dequantizes a packed <see cref="DType.F4E2M1"/> weight <c>[N, K]</c> into the BF16 <c>[N, K]</c> layout <c>IBackend.Linear</c> consumes.</summary>
    /// <param name="blockScale">F8-E4M3 swizzled block scales <c>[paddedRows, paddedCols]</c>; its <see cref="Tensor.Fp8ScaleFactor"/> is part of the arithmetic and must not be dropped.</param>
    /// <param name="globalScale">F32 scalar (a single-element tensor).</param>
    public static Tensor DequantToBf16(Tensor packed, Tensor blockScale, Tensor globalScale)
    {
        ArgumentNullException.ThrowIfNull(packed);
        ArgumentNullException.ThrowIfNull(blockScale);
        ArgumentNullException.ThrowIfNull(globalScale);
        if (packed.DType != DType.F4E2M1 || packed.Shape.Rank != 2)
            throw new ArgumentException($"NVFP4 weight must be F4E2M1 rank-2; got {packed.DType} {packed.Shape}.", nameof(packed));
        if (blockScale.DType != DType.F8E4M3 || blockScale.Shape.Rank != 2)
            throw new ArgumentException($"NVFP4 block scale must be F8E4M3 rank-2; got {blockScale.DType} {blockScale.Shape}.", nameof(blockScale));
        if (globalScale.DType != DType.F32 || globalScale.ElementCount != 1)
            throw new ArgumentException($"NVFP4 global scale must be an F32 scalar; got {globalScale.DType} {globalScale.Shape}.", nameof(globalScale));

        long outDim = packed.Shape[0];
        long inDim = packed.Shape[1];
        if (inDim % GroupSize != 0)
            throw new ArgumentException($"NVFP4 in_features {inDim} must be a multiple of the {GroupSize}-element block.", nameof(packed));
        long paddedCols = blockScale.Shape[1];
        if (blockScale.Shape[0] < outDim || paddedCols < inDim / GroupSize)
            throw new ArgumentException(
                $"NVFP4 block scale {blockScale.Shape} is too small for a [{outDim}, {inDim}] weight.", nameof(blockScale));

        Tensor result = new Tensor(new TensorShape(outDim, inDim), DType.BF16);
        DequantToBf16Core((byte*)packed.DataPointer, (byte*)blockScale.DataPointer,
            blockScale.Fp8ScaleFactor, ((float*)globalScale.DataPointer)[0],
            outDim, inDim / 2, paddedCols, (ushort*)result.DataPointer);
        return result;
    }

    /// <summary>Row-major dequant, parallel over rows: packed reads and BF16 writes are both sequential and the block scale is decoded once per 8 packed bytes.</summary>
    private static void DequantToBf16Core(byte* w, byte* bs, float scaleFactor, float global,
        long outDim, long inHalf, long paddedCols, ushort* dst)
    {
        float[] lut = E2M1Lut;
        float[] e4m3 = E4M3Decode;
        int bytesPerGroup = GroupSize / 2;
        long inDim = inHalf * 2;

        Parallel.For(0, (int)outDim, r =>
        {
            byte* wr = w + (long)r * inHalf;
            ushort* dr = dst + (long)r * inDim;
            float scale = 0f;
            for (long k = 0; k < inHalf; k++)
            {
                if (k % bytesPerGroup == 0)
                    scale = e4m3[bs[SwizzledScaleIndex(r, k / bytesPerGroup, paddedCols)]] * scaleFactor * global;
                byte packed = wr[k];
                dr[2 * k] = TensorCasts.F32ToBf16Bits(lut[(packed >> 4) & 0x0F] * scale);   // high nibble = even element
                dr[2 * k + 1] = TensorCasts.F32ToBf16Bits(lut[packed & 0x0F] * scale);      // low nibble = odd element
            }
        });
    }


    private static float[] BuildE4M3Decode()
    {
        using Tensor bytes = new Tensor(new TensorShape(256), DType.F8E4M3);
        byte* p = (byte*)bytes.DataPointer;
        for (int i = 0; i < 256; i++) p[i] = (byte)i;
        using Tensor decoded = bytes.CastTo(DType.F32);
        return decoded.AsReadOnlySpan<float>().ToArray();
    }
}
