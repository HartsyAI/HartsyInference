using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>IQ3_S (3.4375 bpw i-quant): 256 elements / 110 bytes — <c>[2 bytes FP16 d][64 bytes qs][8 bytes qh][32 bytes signs][4 bytes scales]</c>. A codebook index is 8 bits of qs plus one bit of qh (4 values each), signs are explicit, and a 4-bit scale covers 64 elements. Per ggml <c>dequantize_row_iq3_s</c>: <c>x = d · (1 + 2s) · grid[j] · ±1</c>.</summary>
public sealed unsafe class Codec_IQ3_S : GgufCodecBase
{
    public override DType DType => DType.IQ3_S;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long blocks = (elementCount + 255) / 256;
        for (long sb = 0; sb < blocks; sb++)
        {
            byte* block = src + sb * 110;
            float d = GgufCodecHelpers.ReadHalf(block);
            byte* qs = block + 2;
            byte* qh = block + 66;
            byte* signs = block + 74;
            byte* scales = block + 106;
            float* y = dst + sb * 256;
            long remaining = elementCount - sb * 256;
            for (int ib32 = 0; ib32 < 8; ib32++)
            {
                int s = (ib32 & 1) == 0 ? scales[ib32 / 2] & 0xF : scales[ib32 / 2] >> 4;
                float db = d * (1 + 2 * s);
                byte* q = qs + 8 * ib32;
                byte* sg = signs + 4 * ib32;
                int h = qh[ib32];
                for (int l = 0; l < 4; l++)
                {
                    uint grid1 = IqTables.iq3s_grid[q[2 * l] | ((h << (8 - 2 * l)) & 256)];
                    uint grid2 = IqTables.iq3s_grid[q[2 * l + 1] | ((h << (7 - 2 * l)) & 256)];
                    for (int j = 0; j < 8; j++)
                    {
                        long e = ib32 * 32 + l * 8 + j;
                        uint grid = j < 4 ? grid1 : grid2;
                        if (e < remaining) y[e] = db * (byte)(grid >> (8 * (j & 3))) * ((sg[l] & IqTables.kmask_iq2xs[j]) != 0 ? -1f : 1f);
                    }
                }
            }
        }
    }
}
