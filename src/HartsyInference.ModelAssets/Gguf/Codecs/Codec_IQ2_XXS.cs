using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>IQ2_XXS (2.0625 bpw i-quant): 256 elements / 66 bytes — <c>[2 bytes FP16 d][32 × uint16 qs]</c>. Each 32-element group is two words: the first holds four 8-bit codebook indices, the second four 7-bit sign-pattern indices and a 4-bit scale. Per ggml <c>dequantize_row_iq2_xxs</c>: <c>x = d · (0.5 + s) / 4 · grid[j] · ±1</c>.</summary>
public sealed unsafe class Codec_IQ2_XXS : GgufCodecBase
{
    public override DType DType => DType.IQ2_XXS;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long blocks = (elementCount + 255) / 256;
        for (long sb = 0; sb < blocks; sb++)
        {
            byte* block = src + sb * 66;
            float d = GgufCodecHelpers.ReadHalf(block);
            float* y = dst + sb * 256;
            long remaining = elementCount - sb * 256;
            for (int ib32 = 0; ib32 < 8; ib32++)
            {
                byte* q = block + 2 + 8 * ib32;
                uint aux0 = (uint)(q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24));
                uint aux1 = (uint)(q[4] | (q[5] << 8) | (q[6] << 16) | (q[7] << 24));
                float db = d * (0.5f + (aux1 >> 28)) * 0.25f;
                for (int l = 0; l < 4; l++)
                {
                    ulong grid = IqTables.iq2xxs_grid[(aux0 >> (8 * l)) & 0xFF];
                    byte signs = IqTables.ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                    for (int j = 0; j < 8; j++)
                    {
                        long e = ib32 * 32 + l * 8 + j;
                        if (e < remaining) y[e] = db * (byte)(grid >> (8 * j)) * ((signs & IqTables.kmask_iq2xs[j]) != 0 ? -1f : 1f);
                    }
                }
            }
        }
    }
}
