using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>Q5_K: 5-bit K-quant, 256 elements per super-block, 176 bytes. Layout: <c>[2 d][2 dmin][12 scales+mins][32 high bits][128 low nibbles]</c>. Reconstruction: <c>x[i] = d * sc_j * q[i] - dmin * m_j</c> where <c>q[i] = low | (high &lt;&lt; 4)</c>.</summary>
public sealed unsafe class Codec_Q5_K : GgufCodecBase
{
    public override DType DType => DType.Q5_K;
    public override bool SupportsQuantize => true;

    private const int SuperBlockElems = 256;
    private const int SuperBlockBytes = 176;
    private const int SubBlockElems = 32;
    private const int NumSubBlocks = 8;
    private const int NMax = 31;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long numSuperBlocks = (elementCount + SuperBlockElems - 1) / SuperBlockElems;
        for (long sb = 0; sb < numSuperBlocks; sb++)
        {
            byte* block = src + sb * SuperBlockBytes;
            float d = GgufCodecHelpers.ReadHalf(block);
            float dmin = GgufCodecHelpers.ReadHalf(block + 2);
            byte* scales = block + 4;
            byte* highBits = block + 16;
            byte* lowBits = block + 48;
            long baseIdx = sb * SuperBlockElems;

            for (int j = 0; j < 8; j++)
            {
                GgufCodecHelpers.GetScaleMinK4(j, scales, out byte sc, out byte mm);
                float subScale = d * sc;
                float subMin = dmin * mm;
                byte* subLowBits = lowBits + (j / 2) * SubBlockElems;
                int lowNibbleShift = (j % 2 == 0) ? 0 : 4;
                int highBitShift = j;
                for (int i = 0; i < SubBlockElems; i++)
                {
                    long elemIdx = baseIdx + j * SubBlockElems + i;
                    if (elemIdx >= elementCount) break;
                    int low = (subLowBits[i] >> lowNibbleShift) & 0x0F;
                    int high = ((highBits[i] >> highBitShift) & 0x01) << 4;
                    int q = low | high;
                    dst[elemIdx] = subScale * q - subMin;
                }
            }
        }
    }

    public override void QuantizeFromF32(float* src, byte* dst, long elementCount)
    {
        long numSuperBlocks = (elementCount + SuperBlockElems - 1) / SuperBlockElems;
        byte* L = stackalloc byte[SuperBlockElems];
        float* subScales = stackalloc float[NumSubBlocks];
        float* subMins = stackalloc float[NumSubBlocks];

        for (long sb = 0; sb < numSuperBlocks; sb++)
        {
            byte* block = dst + sb * SuperBlockBytes;
            float* superSrc = src + sb * SuperBlockElems;
            for (int b = 0; b < SuperBlockBytes; b++) block[b] = 0;

            float maxScale = 0f;
            float maxMin = 0f;
            for (int j = 0; j < NumSubBlocks; j++)
            {
                subScales[j] = QkxQuantizer.MakeQkx2Quants(SubBlockElems, NMax,
                    superSrc + j * SubBlockElems, L + j * SubBlockElems, out subMins[j]);
                if (subScales[j] > maxScale) maxScale = subScales[j];
                if (subMins[j] > maxMin) maxMin = subMins[j];
            }

            float invScale = maxScale > 0f ? 63f / maxScale : 0f;
            float invMin = maxMin > 0f ? 63f / maxMin : 0f;
            byte* scalesPacked = block + 4;
            for (int j = 0; j < NumSubBlocks; j++)
            {
                int ls = QkxQuantizer.NearestInt(invScale * subScales[j]);
                int lm = QkxQuantizer.NearestInt(invMin * subMins[j]);
                ls = Math.Clamp(ls, 0, 63);
                lm = Math.Clamp(lm, 0, 63);
                QkxQuantizer.PackScaleMinK4((byte)ls, (byte)lm, j, scalesPacked);
            }
            *(Half*)block = (Half)(maxScale / 63f);
            *(Half*)(block + 2) = (Half)(maxMin / 63f);

            float d = (float)(*(Half*)block);
            float dmin = (float)(*(Half*)(block + 2));
            for (int j = 0; j < NumSubBlocks; j++)
            {
                GgufCodecHelpers.GetScaleMinK4(j, scalesPacked, out byte scq, out byte mmq);
                float fd = d * scq;
                if (fd == 0f)
                {
                    for (int i = 0; i < SubBlockElems; i++) L[j * SubBlockElems + i] = 0;
                    continue;
                }
                float fdm = dmin * mmq;
                for (int i = 0; i < SubBlockElems; i++)
                {
                    long elemIdx = sb * SuperBlockElems + j * SubBlockElems + i;
                    float xv = (elemIdx < elementCount) ? superSrc[j * SubBlockElems + i] : 0f;
                    int l = QkxQuantizer.NearestInt((xv + fdm) / fd);
                    L[j * SubBlockElems + i] = (byte)Math.Clamp(l, 0, 31);
                }
            }

            byte* highOut = block + 16;
            byte* lowOut = block + 48;
            for (int j = 0; j < SuperBlockElems; j += 64)
            {
                int subBlockA = j / 32;
                int subBlockB = subBlockA + 1;
                int shiftA = subBlockA;
                int shiftB = subBlockB;
                for (int l = 0; l < 32; l++)
                {
                    int qA = L[j + l];
                    int qB = L[j + l + 32];
                    lowOut[(j / 64) * 32 + l] = (byte)((qA & 0x0F) | ((qB & 0x0F) << 4));
                    highOut[l] = (byte)(highOut[l] | (((qA >> 4) & 0x01) << shiftA));
                    highOut[l] = (byte)(highOut[l] | (((qB >> 4) & 0x01) << shiftB));
                }
            }
        }
    }
}
