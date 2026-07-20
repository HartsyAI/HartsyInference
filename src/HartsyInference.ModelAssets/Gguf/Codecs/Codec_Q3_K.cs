using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>Q3_K: 3-bit K-quant, 256 elements / 110 bytes. Layout (canonical ggml `block_q3_K`):
/// <list type="bullet">
/// <item>32 bytes hmask — high bit (bit 2) of each 3-bit value, 256 bits</item>
/// <item>64 bytes qs — low 2 bits of each 3-bit value, packed 4 per byte</item>
/// <item>12 bytes scales — 6-bit signed scales (range -32..31), 16 sub-blocks packed</item>
/// <item>2 bytes FP16 d (super-block scale)</item>
/// </list>
/// Reconstruction (per ggml `dequantize_row_q3_K`): 256 elements in 2 halves of 128. Each iteration processes 32 elements from <c>qs</c> + corresponding hmask bits. The 6-bit signed scale is unpacked from the 12-byte scales array; <c>q = ((qs &gt;&gt; shift) &amp; 3) - ((hmask_bit) ? 0 : 4)</c>, then <c>x = d * scale * q</c>.</summary>
public sealed unsafe class Codec_Q3_K : GgufCodecBase
{
    public override DType DType => DType.Q3_K;

    private const int SuperBlockElems = 256;
    private const int SuperBlockBytes = 110;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long numSuperBlocks = (elementCount + SuperBlockElems - 1) / SuperBlockElems;
        sbyte* aux = stackalloc sbyte[16];

        for (long sb = 0; sb < numSuperBlocks; sb++)
        {
            byte* block = src + sb * SuperBlockBytes;
            byte* hmask = block;
            byte* qs = block + 32;
            byte* scalesPacked = block + 96;
            float d = GgufCodecHelpers.ReadHalf(block + 108);

            uint mask = 1;
            UnpackQ3KScales(scalesPacked, aux);
            long baseElem = sb * SuperBlockElems;
            int writeIdx = 0;
            int scaleIdx = 0;

            for (int n = 0; n < SuperBlockElems; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    float dl1 = d * aux[scaleIdx];
                    for (int l = 0; l < 16; l++)
                    {
                        long elemIdx = baseElem + writeIdx;
                        if (elemIdx < elementCount)
                        {
                            int q = (qs[l] >> shift) & 0x03;
                            int hbit = (hmask[l] & mask) != 0 ? 0 : 4;
                            dst[elemIdx] = dl1 * (q - hbit);
                        }
                        writeIdx++;
                    }
                    scaleIdx++;
                    float dl2 = d * aux[scaleIdx];
                    for (int l = 0; l < 16; l++)
                    {
                        long elemIdx = baseElem + writeIdx;
                        if (elemIdx < elementCount)
                        {
                            int q = (qs[l + 16] >> shift) & 0x03;
                            int hbit = (hmask[l + 16] & mask) != 0 ? 0 : 4;
                            dst[elemIdx] = dl2 * (q - hbit);
                        }
                        writeIdx++;
                    }
                    scaleIdx++;
                    shift += 2;
                    mask <<= 1;
                }
                qs += 32;
            }
        }
    }

    /// <summary>Unpacks 16 6-bit signed scales from 12 bytes per the canonical ggml layout. Bytes 0..7 contain the low 4 bits of each scale plus packed high 2 bits in bytes 8..11.</summary>
    private static void UnpackQ3KScales(byte* packed, sbyte* result)
    {
        for (int i = 0; i < 8; i++)
        {
            byte lowByte = packed[i];
            int low0 = lowByte & 0x0F;
            int low1 = lowByte >> 4;
            byte highByte = packed[8 + i / 2];
            int hi0 = (i % 2 == 0) ? (highByte & 0x03) : ((highByte >> 4) & 0x03);
            int hi1 = (i % 2 == 0) ? ((highByte >> 2) & 0x03) : ((highByte >> 6) & 0x03);
            int s0 = low0 | (hi0 << 4);
            int s1 = low1 | (hi1 << 4);
            result[2 * i + 0] = (sbyte)(s0 - 32);
            result[2 * i + 1] = (sbyte)(s1 - 32);
        }
    }
}
