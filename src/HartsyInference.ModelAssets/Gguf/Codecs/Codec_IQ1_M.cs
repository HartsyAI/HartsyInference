using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>IQ1_M (1.75 bpw i-quant): 256 elements / 56 bytes — <c>[32 bytes qs][16 bytes qh][8 bytes scales]</c>, no separate d: the FP16 super-scale is spread over the four scale words' top nibbles. A codebook index is 8 bits of qs plus 3 bits of a qh nibble whose top bit picks the ±1/8 shift; 3-bit scales cover 16 elements. Per ggml <c>dequantize_row_iq1_m</c>: <c>x = d · (2s + 1) · (grid[j] ± 1/8)</c>.</summary>
public sealed unsafe class Codec_IQ1_M : GgufCodecBase
{
    public override DType DType => DType.IQ1_M;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long blocks = (elementCount + 255) / 256;
        for (long sb = 0; sb < blocks; sb++)
        {
            byte* block = src + sb * 56;
            byte* qs = block;
            byte* qh = block + 32;
            byte* scb = block + 48;
            int sc0 = scb[0] | (scb[1] << 8), sc1 = scb[2] | (scb[3] << 8), sc2 = scb[4] | (scb[5] << 8), sc3 = scb[6] | (scb[7] << 8);
            ushort dBits = (ushort)((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000));
            float d = (float)BitConverter.UInt16BitsToHalf(dBits);
            float* y = dst + sb * 256;
            long remaining = elementCount - sb * 256;
            for (int ib = 0; ib < 8; ib++)
            {
                int sc = (ib / 2) switch { 0 => sc0, 1 => sc1, 2 => sc2, _ => sc3 };
                float dl1 = d * (2 * ((sc >> (6 * (ib % 2) + 0)) & 7) + 1);
                float dl2 = d * (2 * ((sc >> (6 * (ib % 2) + 3)) & 7) + 1);
                byte* q = qs + 4 * ib;
                byte* h = qh + 2 * ib;
                for (int l = 0; l < 4; l++)
                {
                    int hb = h[l >> 1];
                    int idx = q[l] | (((l & 1) == 0 ? hb << 8 : hb << 4) & 0x700);
                    float delta = (hb & ((l & 1) == 0 ? 0x08 : 0x80)) != 0 ? -Codec_IQ1_S.Delta : Codec_IQ1_S.Delta;
                    float dl = l < 2 ? dl1 : dl2;
                    ulong grid = IqTables.iq1s_grid[idx];
                    for (int j = 0; j < 8; j++)
                    {
                        long e = ib * 32 + l * 8 + j;
                        if (e < remaining) y[e] = dl * ((sbyte)(byte)(grid >> (8 * j)) + delta);
                    }
                }
            }
        }
    }
}
