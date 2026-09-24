using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>IQ4_XS (4-bit non-linear i-quant, super-blocked): 256 elements / 136 bytes — <c>[2 bytes FP16 d][2 bytes scales_h][4 bytes scales_l][128 bytes nibbles]</c>, canonical ggml <c>block_iq4_xs</c>. Eight 32-element sub-blocks each carry a 6-bit scale split across <c>scales_l</c> (low 4 bits, one nibble per sub-block) and <c>scales_h</c> (high 2 bits, two per sub-block), offset by 32. Nibbles index the same <c>kvalues_iq4nl</c> table as <see cref="Codec_IQ4_NL"/>, low nibbles for elements 0..15 of a sub-block and high nibbles for 16..31.
///
/// <para>Reconstruction per ggml <c>dequantize_row_iq4_xs</c>: <c>x = d · (ls − 32) · kvalues[q]</c>. The most widely published i-quant for consumer LLM builds; the loader used to refuse it as "no codec registered".</para></summary>
public sealed unsafe class Codec_IQ4_XS : GgufCodecBase
{
    public override DType DType => DType.IQ4_XS;

    private const int SuperBlockElems = 256;
    private const int SuperBlockBytes = 136;
    private const int SubElems = 32;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long numSuperBlocks = (elementCount + SuperBlockElems - 1) / SuperBlockElems;
        fixed (sbyte* lookup = Codec_IQ4_NL.KValues)
        {
            for (long sb = 0; sb < numSuperBlocks; sb++)
            {
                byte* block = src + sb * SuperBlockBytes;
                float d = GgufCodecHelpers.ReadHalf(block);
                int scalesH = block[2] | (block[3] << 8);
                byte* scalesL = block + 4;
                byte* qs = block + 8;
                long baseElem = sb * SuperBlockElems;
                for (int ib = 0; ib < SuperBlockElems / SubElems; ib++)
                {
                    int ls = ((scalesL[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scalesH >> (2 * ib)) & 3) << 4);
                    float dl = d * (ls - 32);
                    byte* q = qs + ib * 16;
                    long e = baseElem + ib * SubElems;
                    for (int j = 0; j < 16; j++)
                    {
                        if (e + j < elementCount) dst[e + j] = dl * lookup[q[j] & 0xF];
                        if (e + 16 + j < elementCount) dst[e + 16 + j] = dl * lookup[q[j] >> 4];
                    }
                }
            }
        }
    }
}
