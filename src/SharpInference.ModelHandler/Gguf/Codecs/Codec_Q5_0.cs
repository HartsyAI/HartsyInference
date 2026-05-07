using SharpInference.Core.Tensors;

namespace SharpInference.ModelHandler.Gguf.Codecs;

/// <summary>Q5_0: 5-bit, 32 elements / 22 bytes. Layout: <c>[2 scale][4 high bits (32 bits packed LE)][16 low nibbles]</c>. Reconstruction: <c>x[i] = scale * (q[i] - 16)</c> where <c>q[i] = low | (high&lt;&lt;4) ∈ [0..31]</c>. The low nibble for element i comes from byte i of the 16-byte nibble block (low half) or i+16 (high half), and the high bit is bit i of the 32-bit packed word.</summary>
public sealed unsafe class Codec_Q5_0 : GgufCodecBase
{
    public override DType DType => DType.Q5_0;

    private const int BlockElems = 32;
    private const int BlockBytes = 22;
    private const int HalfBlock = 16;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long numBlocks = (elementCount + BlockElems - 1) / BlockElems;
        for (long b = 0; b < numBlocks; b++)
        {
            byte* block = src + b * BlockBytes;
            float scale = GgufCodecHelpers.ReadHalf(block);
            uint highBitsPacked = *(uint*)(block + 2);
            byte* q = block + 6;
            long baseIdx = b * BlockElems;
            for (int i = 0; i < HalfBlock; i++)
            {
                long elemIdx0 = baseIdx + i;
                long elemIdx1 = baseIdx + i + HalfBlock;
                int xh0 = (int)((highBitsPacked >> i) & 0x01) << 4;
                int xh1 = (int)((highBitsPacked >> (i + HalfBlock)) & 0x01) << 4;
                int qLow0 = (q[i] & 0x0F) | xh0;
                int qLow1 = ((q[i] >> 4) & 0x0F) | xh1;
                if (elemIdx0 < elementCount) dst[elemIdx0] = scale * (qLow0 - 16);
                if (elemIdx1 < elementCount) dst[elemIdx1] = scale * (qLow1 - 16);
            }
        }
    }
}
