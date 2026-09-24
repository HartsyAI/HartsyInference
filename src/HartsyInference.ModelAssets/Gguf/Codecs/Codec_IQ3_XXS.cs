using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>IQ3_XXS (3.0625 bpw i-quant): 256 elements / 98 bytes — <c>[2 bytes FP16 d][64 bytes qs][32 bytes scales-and-signs]</c>. Each 32-element group is eight 8-bit codebook indices (4 values each) plus a word of four 7-bit sign patterns and a 4-bit scale. Per ggml <c>dequantize_row_iq3_xxs</c>: <c>x = d · (0.5 + s) / 2 · grid[j] · ±1</c>.</summary>
public sealed unsafe class Codec_IQ3_XXS : GgufCodecBase
{
    public override DType DType => DType.IQ3_XXS;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long blocks = (elementCount + 255) / 256;
        for (long sb = 0; sb < blocks; sb++)
        {
            byte* block = src + sb * 98;
            float d = GgufCodecHelpers.ReadHalf(block);
            byte* qs = block + 2;
            byte* ss = block + 2 + 64;
            float* y = dst + sb * 256;
            long remaining = elementCount - sb * 256;
            for (int ib32 = 0; ib32 < 8; ib32++)
            {
                byte* w = ss + 4 * ib32;
                uint aux = (uint)(w[0] | (w[1] << 8) | (w[2] << 16) | (w[3] << 24));
                float db = d * (0.5f + (aux >> 28)) * 0.5f;
                byte* q = qs + 8 * ib32;
                for (int l = 0; l < 4; l++)
                {
                    byte signs = IqTables.ksigns_iq2xs[(aux >> (7 * l)) & 127];
                    uint grid1 = IqTables.iq3xxs_grid[q[2 * l]];
                    uint grid2 = IqTables.iq3xxs_grid[q[2 * l + 1]];
                    for (int j = 0; j < 8; j++)
                    {
                        long e = ib32 * 32 + l * 8 + j;
                        uint grid = j < 4 ? grid1 : grid2;
                        if (e < remaining) y[e] = db * (byte)(grid >> (8 * (j & 3))) * ((signs & IqTables.kmask_iq2xs[j]) != 0 ? -1f : 1f);
                    }
                }
            }
        }
    }
}
