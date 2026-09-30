using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>IQ2_S (2.5625 bpw i-quant): 256 elements / 82 bytes — <c>[2 bytes FP16 d][32 bytes qs][32 bytes signs][8 bytes qh][8 bytes scales]</c>. A codebook index is 8 bits of qs plus 2 bits from qh; signs are one explicit byte per 8 elements. Per ggml <c>dequantize_row_iq2_s</c>: <c>x = d · (0.5 + s) / 4 · grid[j] · ±1</c>.</summary>
public sealed unsafe class Codec_IQ2_S : GgufCodecBase
{
    public override DType DType => DType.IQ2_S;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long blocks = (elementCount + 255) / 256;
        for (long sb = 0; sb < blocks; sb++)
        {
            byte* block = src + sb * 82;
            float d = GgufCodecHelpers.ReadHalf(block);
            byte* qs = block + 2;
            byte* signs = block + 2 + 32;
            byte* qh = block + 66;
            byte* scales = block + 74;
            float* y = dst + sb * 256;
            long remaining = elementCount - sb * 256;
            for (int ib32 = 0; ib32 < 8; ib32++)
            {
                float db0 = d * (0.5f + (scales[ib32] & 0xF)) * 0.25f;
                float db1 = d * (0.5f + (scales[ib32] >> 4)) * 0.25f;
                for (int l = 0; l < 4; l++)
                {
                    ulong grid = IqTables.iq2s_grid[qs[4 * ib32 + l] | ((qh[ib32] << (8 - 2 * l)) & 0x300)];
                    byte sg = signs[4 * ib32 + l];
                    float dl = l < 2 ? db0 : db1;
                    for (int j = 0; j < 8; j++)
                    {
                        long e = ib32 * 32 + l * 8 + j;
                        if (e < remaining) y[e] = dl * (byte)(grid >> (8 * j)) * ((sg & IqTables.kmask_iq2xs[j]) != 0 ? -1f : 1f);
                    }
                }
            }
        }
    }
}
