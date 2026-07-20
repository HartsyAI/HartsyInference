using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>Q4_1: 4-bit + min, 32 elements / 20 bytes. Layout: <c>[2 scale][2 min][16 nibbles]</c>. Reconstruction: <c>x[i] = scale * q[i] + min</c> where <c>q[i] ∈ [0..15]</c>.</summary>
public sealed unsafe class Codec_Q4_1 : GgufCodecBase
{
    public override DType DType => DType.Q4_1;

    private const int BlockElems = 32;
    private const int BlockBytes = 20;
    private const int HalfBlock = 16;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long numBlocks = (elementCount + BlockElems - 1) / BlockElems;
        for (long b = 0; b < numBlocks; b++)
        {
            byte* block = src + b * BlockBytes;
            float scale = GgufCodecHelpers.ReadHalf(block);
            float min = GgufCodecHelpers.ReadHalf(block + 2);
            byte* q = block + 4;
            long baseIdx = b * BlockElems;
            for (int i = 0; i < HalfBlock; i++)
            {
                long elemIdx0 = baseIdx + i;
                long elemIdx1 = baseIdx + i + HalfBlock;
                int qLow = q[i] & 0x0F;
                int qHigh = (q[i] >> 4) & 0x0F;
                if (elemIdx0 < elementCount) dst[elemIdx0] = scale * qLow + min;
                if (elemIdx1 < elementCount) dst[elemIdx1] = scale * qHigh + min;
            }
        }
    }
}
