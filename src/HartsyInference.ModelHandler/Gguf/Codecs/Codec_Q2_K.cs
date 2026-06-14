using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelHandler.Gguf.Codecs;

/// <summary>Q2_K: 2-bit K-quant, 256 elements / 84 bytes. Layout (canonical ggml `block_q2_K`):
/// <list type="bullet">
/// <item>16 bytes scales — 4-bit scale (low nibble) + 4-bit min (high nibble) per 16-element sub-block</item>
/// <item>64 bytes qs — 2-bit packed quants</item>
/// <item>2 bytes FP16 d (super-block scale)</item>
/// <item>2 bytes FP16 dmin (super-block min)</item>
/// </list>
/// Reconstruction per ggml `dequantize_row_q2_K`: 256 elements processed in 2 halves of 128. Each half consumes 32 bytes of qs and 8 scale entries, with 4 shift positions (0,2,4,6) extracting the four 2-bit pairs. <c>x = d * (scale &amp; 0xF) * q - dmin * (scale &gt;&gt; 4)</c> where <c>q ∈ [0..3]</c>.
///
/// <para><b>Aggressive quant — quality cost is significant.</b> Mostly used for LLMs where memory beats fidelity; rare for diffusion. Implementation provided for completeness.</para></summary>
public sealed unsafe class Codec_Q2_K : GgufCodecBase
{
    public override DType DType => DType.Q2_K;

    private const int SuperBlockElems = 256;
    private const int SuperBlockBytes = 84;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long numSuperBlocks = (elementCount + SuperBlockElems - 1) / SuperBlockElems;
        for (long sb = 0; sb < numSuperBlocks; sb++)
        {
            byte* block = src + sb * SuperBlockBytes;
            byte* scales = block;
            byte* qs = block + 16;
            float d = GgufCodecHelpers.ReadHalf(block + 80);
            float min = GgufCodecHelpers.ReadHalf(block + 82);
            long baseElem = sb * SuperBlockElems;
            int writeIdx = 0;
            int isCounter = 0;

            for (int n = 0; n < SuperBlockElems; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    byte sc1 = scales[isCounter++];
                    float dl1 = d * (sc1 & 0x0F);
                    float ml1 = min * (sc1 >> 4);
                    for (int l = 0; l < 16; l++)
                    {
                        long elemIdx = baseElem + writeIdx;
                        if (elemIdx < elementCount)
                        {
                            int q = (qs[l] >> shift) & 0x03;
                            dst[elemIdx] = dl1 * q - ml1;
                        }
                        writeIdx++;
                    }
                    byte sc2 = scales[isCounter++];
                    float dl2 = d * (sc2 & 0x0F);
                    float ml2 = min * (sc2 >> 4);
                    for (int l = 0; l < 16; l++)
                    {
                        long elemIdx = baseElem + writeIdx;
                        if (elemIdx < elementCount)
                        {
                            int q = (qs[l + 16] >> shift) & 0x03;
                            dst[elemIdx] = dl2 * q - ml2;
                        }
                        writeIdx++;
                    }
                    shift += 2;
                }
                qs += 32;
            }
        }
    }
}
