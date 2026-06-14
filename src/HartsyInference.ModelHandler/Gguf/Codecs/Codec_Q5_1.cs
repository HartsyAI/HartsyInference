using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelHandler.Gguf.Codecs;

/// <summary>Q5_1: 5-bit + min, 32 elements / 24 bytes. Layout: <c>[2 scale][2 min][4 high bits][16 low nibbles]</c>. Reconstruction: <c>x[i] = scale * q[i] + min</c> where <c>q[i] ∈ [0..31]</c>.</summary>
public sealed unsafe class Codec_Q5_1 : GgufCodecBase
{
    public override DType DType => DType.Q5_1;

    private const int BlockElems = 32;
    private const int BlockBytes = 24;
    private const int HalfBlock = 16;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long numBlocks = (elementCount + BlockElems - 1) / BlockElems;
        for (long b = 0; b < numBlocks; b++)
        {
            byte* block = src + b * BlockBytes;
            float scale = GgufCodecHelpers.ReadHalf(block);
            float min = GgufCodecHelpers.ReadHalf(block + 2);
            uint highBitsPacked = *(uint*)(block + 4);
            byte* q = block + 8;
            long baseIdx = b * BlockElems;
            for (int i = 0; i < HalfBlock; i++)
            {
                long elemIdx0 = baseIdx + i;
                long elemIdx1 = baseIdx + i + HalfBlock;
                int xh0 = (int)((highBitsPacked >> i) & 0x01) << 4;
                int xh1 = (int)((highBitsPacked >> (i + HalfBlock)) & 0x01) << 4;
                int qLow0 = (q[i] & 0x0F) | xh0;
                int qLow1 = ((q[i] >> 4) & 0x0F) | xh1;
                if (elemIdx0 < elementCount) dst[elemIdx0] = scale * qLow0 + min;
                if (elemIdx1 < elementCount) dst[elemIdx1] = scale * qLow1 + min;
            }
        }
    }
}
