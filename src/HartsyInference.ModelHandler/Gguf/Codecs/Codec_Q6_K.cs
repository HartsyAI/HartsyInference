using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelHandler.Gguf.Codecs;

/// <summary>Q6_K: 6-bit K-quant, 256 elements / 210 bytes super-block. Highest-quality k-quant, often used for output-projection weights in `_M` mix policies. Layout (canonical ggml `block_q6_K`): <c>[128 bytes ql (low 4 bits per element)][64 bytes qh (high 2 bits per element, packed 4-per-byte)][16 bytes int8 scales (one per 16-element sub-block)][2 bytes FP16 d (super-block scale)]</c>.
///
/// <para>Reconstruction (per ggml `dequantize_row_q6_K`): for sub-block <c>j ∈ [0..15]</c> covering 16 elements <c>i = j*16 .. j*16+15</c>: <c>q = (ql_low_or_high) | (qh_2bit &lt;&lt; 4) - 32</c>, then <c>x = d * scales[j] * q</c>. The `qh` 2-bit values are arranged as 4 groups of 16 covering the 4 sets of 32 contiguous elements.</para></summary>
public sealed unsafe class Codec_Q6_K : GgufCodecBase
{
    public override DType DType => DType.Q6_K;
    public override bool SupportsQuantize => true;

    private const int SuperBlockElems = 256;
    private const int SuperBlockBytes = 210;
    private const int SubBlockElems = 16;
    private const int NumSubBlocks = 16;
    private const int NMax = 32;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long numSuperBlocks = (elementCount + SuperBlockElems - 1) / SuperBlockElems;
        for (long sb = 0; sb < numSuperBlocks; sb++)
        {
            byte* block = src + sb * SuperBlockBytes;
            byte* ql = block;
            byte* qh = block + 128;
            sbyte* scales = (sbyte*)(block + 192);
            float d = GgufCodecHelpers.ReadHalf(block + 208);
            long baseIdx = sb * SuperBlockElems;

            // Canonical ggml unrolling: process the 256 elements in 2 halves of 128 each.
            // Each half uses 64 bytes of ql, 32 bytes of qh, 8 sub-blocks of 16 elements.
            for (int half = 0; half < 2; half++)
            {
                byte* qlH = ql + half * 64;
                byte* qhH = qh + half * 32;
                sbyte* scH = scales + half * 8;
                int halfBaseElem = half * 128;

                for (int l = 0; l < 32; l++)
                {
                    int isOffset = l / 16;
                    int q1 = ((qlH[l] & 0x0F) | (((qhH[l] >> 0) & 0x03) << 4)) - 32;
                    int q2 = ((qlH[l + 32] & 0x0F) | (((qhH[l] >> 2) & 0x03) << 4)) - 32;
                    int q3 = ((qlH[l] >> 4) | (((qhH[l] >> 4) & 0x03) << 4)) - 32;
                    int q4 = ((qlH[l + 32] >> 4) | (((qhH[l] >> 6) & 0x03) << 4)) - 32;

                    long e1 = baseIdx + halfBaseElem + l;
                    long e2 = baseIdx + halfBaseElem + l + 32;
                    long e3 = baseIdx + halfBaseElem + l + 64;
                    long e4 = baseIdx + halfBaseElem + l + 96;

                    if (e1 < elementCount) dst[e1] = d * scH[isOffset + 0] * q1;
                    if (e2 < elementCount) dst[e2] = d * scH[isOffset + 2] * q2;
                    if (e3 < elementCount) dst[e3] = d * scH[isOffset + 4] * q3;
                    if (e4 < elementCount) dst[e4] = d * scH[isOffset + 6] * q4;
                }
            }
        }
    }

    public override void QuantizeFromF32(float* src, byte* dst, long elementCount)
    {
        long numSuperBlocks = (elementCount + SuperBlockElems - 1) / SuperBlockElems;
        sbyte* L = stackalloc sbyte[SuperBlockElems];
        float* subScales = stackalloc float[NumSubBlocks];

        for (long sb = 0; sb < numSuperBlocks; sb++)
        {
            byte* block = dst + sb * SuperBlockBytes;
            float* superSrc = src + sb * SuperBlockElems;
            for (int b = 0; b < SuperBlockBytes; b++) block[b] = 0;

            float maxAbsScale = 0f;
            for (int j = 0; j < NumSubBlocks; j++)
            {
                subScales[j] = QkxQuantizer.MakeSymmetricScale(SubBlockElems, NMax,
                    superSrc + j * SubBlockElems, L + j * SubBlockElems);
                if (MathF.Abs(subScales[j]) > maxAbsScale) maxAbsScale = MathF.Abs(subScales[j]);
            }

            float invScale = maxAbsScale > 0f ? 127f / maxAbsScale : 0f;
            sbyte* scalesOut = (sbyte*)(block + 192);
            for (int j = 0; j < NumSubBlocks; j++)
            {
                int s = QkxQuantizer.NearestInt(subScales[j] * invScale);
                scalesOut[j] = (sbyte)Math.Clamp(s, -127, 127);
            }
            *(Half*)(block + 208) = (Half)(maxAbsScale / 127f);

            float d = (float)(*(Half*)(block + 208));
            for (int j = 0; j < NumSubBlocks; j++)
            {
                float subD = d * scalesOut[j];
                if (subD == 0f)
                {
                    for (int i = 0; i < SubBlockElems; i++) L[j * SubBlockElems + i] = 0;
                    continue;
                }
                float invSubD = 1f / subD;
                for (int i = 0; i < SubBlockElems; i++)
                {
                    long elemIdx = sb * SuperBlockElems + j * SubBlockElems + i;
                    float xv = (elemIdx < elementCount) ? superSrc[j * SubBlockElems + i] : 0f;
                    int q = QkxQuantizer.NearestInt(xv * invSubD);
                    L[j * SubBlockElems + i] = (sbyte)Math.Clamp(q + 32, 0, 63);
                }
            }

            byte* qlOut = block;
            byte* qhOut = block + 128;
            for (int half = 0; half < 2; half++)
            {
                byte* qlH = qlOut + half * 64;
                byte* qhH = qhOut + half * 32;
                int halfBaseElem = half * 128;
                for (int l = 0; l < 32; l++)
                {
                    int v0 = (byte)L[halfBaseElem + l];
                    int v1 = (byte)L[halfBaseElem + l + 32];
                    int v2 = (byte)L[halfBaseElem + l + 64];
                    int v3 = (byte)L[halfBaseElem + l + 96];
                    qlH[l] = (byte)((v0 & 0x0F) | ((v2 & 0x0F) << 4));
                    qlH[l + 32] = (byte)((v1 & 0x0F) | ((v3 & 0x0F) << 4));
                    qhH[l] = (byte)(((v0 >> 4) & 0x03) | (((v1 >> 4) & 0x03) << 2) | (((v2 >> 4) & 0x03) << 4) | (((v3 >> 4) & 0x03) << 6));
                }
            }
        }
    }
}
