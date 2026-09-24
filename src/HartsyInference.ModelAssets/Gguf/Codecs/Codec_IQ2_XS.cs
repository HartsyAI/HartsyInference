using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>IQ2_XS (2.3125 bpw i-quant): 256 elements / 74 bytes — <c>[2 bytes FP16 d][32 × uint16 qs][8 bytes scales]</c>. A word is a 9-bit codebook index and a 7-bit sign pattern; each 32-element group has two 4-bit scales. Per ggml <c>dequantize_row_iq2_xs</c>: <c>x = d · (0.5 + s) / 4 · grid[j] · ±1</c>.</summary>
public sealed unsafe class Codec_IQ2_XS : GgufCodecBase
{
    public override DType DType => DType.IQ2_XS;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long blocks = (elementCount + 255) / 256;
        for (long sb = 0; sb < blocks; sb++)
        {
            byte* block = src + sb * 74;
            float d = GgufCodecHelpers.ReadHalf(block);
            byte* scales = block + 66;
            float* y = dst + sb * 256;
            long remaining = elementCount - sb * 256;
            for (int ib32 = 0; ib32 < 8; ib32++)
            {
                float db0 = d * (0.5f + (scales[ib32] & 0xF)) * 0.25f;
                float db1 = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
                for (int l = 0; l < 4; l++)
                {
                    byte* q = block + 2 + 2 * (4 * ib32 + l);
                    int word = q[0] | (q[1] << 8);
                    ulong grid = IqTables.iq2xs_grid[word & 511];
                    byte signs = IqTables.ksigns_iq2xs[word >> 9];
                    float dl = l < 2 ? db0 : db1;
                    for (int j = 0; j < 8; j++)
                    {
                        long e = ib32 * 32 + l * 8 + j;
                        if (e < remaining) y[e] = dl * (byte)(grid >> (8 * j)) * ((signs & IqTables.kmask_iq2xs[j]) != 0 ? -1f : 1f);
                    }
                }
            }
        }
    }
}
