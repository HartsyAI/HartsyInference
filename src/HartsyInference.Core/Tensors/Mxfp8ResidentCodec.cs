namespace HartsyInference.Core.Tensors;

/// <summary>Host decode of a resident MXFP8 weight — an F8E4M3 <c>[N, K]</c> matrix with one UE8M0 (power-of-two) scale per 32 input elements in NVIDIA's blocked layout, the ComfyUI <c>mxfp8</c> packaging. Lives beside <see cref="Nvfp4ResidentCodec"/> so a backend that cannot serve the packed weight can still unpack it without a dependency on the model-assets package.</summary>
public static unsafe class Mxfp8ResidentCodec
{
    /// <summary>Input elements per block scale.</summary>
    public const int GroupSize = 32;

    /// <summary>UE8M0 byte → float: <c>2^(byte − 127)</c>, and 0 for byte 0 — the bit reinterpretation <c>byte &lt;&lt; 23</c>.</summary>
    public static float E8M0Decode(byte e8m0) => BitConverter.UInt32BitsToSingle((uint)e8m0 << 23);

    /// <summary>Dequantizes <paramref name="weight"/> (F8E4M3 <c>[N, K]</c>) against <paramref name="blockScale"/> (U8 <c>[≥N padded to 128, ≥K/32 padded to 4]</c>, blocked layout) to BF16 <c>[N, K]</c>.</summary>
    public static Tensor DequantToBf16(Tensor weight, Tensor blockScale)
    {
        ArgumentNullException.ThrowIfNull(weight);
        ArgumentNullException.ThrowIfNull(blockScale);
        if (weight.DType != DType.F8E4M3 || weight.Shape.Rank != 2)
            throw new ArgumentException($"MXFP8 weight must be F8E4M3 [N, K]; got {weight.DType} {weight.Shape}.", nameof(weight));
        if (blockScale.DType != DType.U8 || blockScale.Shape.Rank != 2)
            throw new ArgumentException($"MXFP8 block scale must be U8 rank-2; got {blockScale.DType} {blockScale.Shape}.", nameof(blockScale));

        long n = weight.Shape[0], k = weight.Shape[1];
        long paddedCols = blockScale.Shape[1];
        using Tensor wF32 = weight.CastTo(DType.F32);
        using Tensor outF32 = new Tensor(new TensorShape(n, k), DType.F32);
        float* w = (float*)wF32.DataPointer;
        byte* s = (byte*)blockScale.DataPointer;
        float* o = (float*)outF32.DataPointer;
        float factor = blockScale.Fp8ScaleFactor;   // the GPU unpack applies it too; 1.0 on every loader today
        for (long r = 0; r < n; r++)
        {
            for (long c = 0; c < k; c++)
            {
                float scale = E8M0Decode(s[Nvfp4ResidentCodec.SwizzledScaleIndex(r, c / GroupSize, paddedCols)]);
                o[r * k + c] = w[r * k + c] * scale * factor;
            }
        }
        return outF32.CastTo(DType.BF16);
    }
}
