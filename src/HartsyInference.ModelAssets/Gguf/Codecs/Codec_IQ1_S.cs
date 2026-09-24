using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>IQ1_S (1.5625 bpw i-quant): 256 elements / 50 bytes — <c>[2 bytes FP16 d][32 bytes qs][8 × uint16 qh]</c>. A codebook index is 8 bits of qs plus 3 bits of qh (8 signed values each); each 32-element group carries a 3-bit scale and a sign bit selecting the ±1/8 shift. Per ggml <c>dequantize_row_iq1_s</c>: <c>x = d · (2s + 1) · (grid[j] ± 1/8)</c>.</summary>
public sealed unsafe class Codec_IQ1_S : GgufCodecBase
{
    public override DType DType => DType.IQ1_S;

    internal const float Delta = 0.125f;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long blocks = (elementCount + 255) / 256;
        for (long sb = 0; sb < blocks; sb++)
        {
            byte* block = src + sb * 50;
            float d = GgufCodecHelpers.ReadHalf(block);
            byte* qs = block + 2;
            byte* qhb = block + 34;
            float* y = dst + sb * 256;
            long remaining = elementCount - sb * 256;
            for (int ib = 0; ib < 8; ib++)
            {
                int qh = qhb[2 * ib] | (qhb[2 * ib + 1] << 8);
                float dl = d * (2 * ((qh >> 12) & 7) + 1);
                float delta = (qh & 0x8000) != 0 ? -Delta : Delta;
                for (int l = 0; l < 4; l++)
                {
                    ulong grid = IqTables.iq1s_grid[qs[4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)];
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
