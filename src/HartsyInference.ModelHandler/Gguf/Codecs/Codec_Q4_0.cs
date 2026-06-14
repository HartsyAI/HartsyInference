using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelHandler.Gguf.Codecs;

/// <summary>Q4_0: 4-bit, 32 elements per block, 18 bytes. Layout: <c>[2 bytes FP16 scale][16 bytes nibbles]</c>. Reconstruction: <c>x[i] = scale * (q[i] - 8)</c> where <c>q[i] ∈ [0..15]</c>. The first 16 elements use low nibbles, the last 16 use high nibbles (canonical ggml `dequantize_row_q4_0`).</summary>
public sealed unsafe class Codec_Q4_0 : GgufCodecBase
{
    public override DType DType => DType.Q4_0;

    private const int BlockElems = 32;
    private const int BlockBytes = 18;
    private const int HalfBlock = 16;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long numBlocks = (elementCount + BlockElems - 1) / BlockElems;
        for (long b = 0; b < numBlocks; b++)
        {
            byte* block = src + b * BlockBytes;
            float scale = GgufCodecHelpers.ReadHalf(block);
            byte* q = block + 2;
            long baseIdx = b * BlockElems;
            for (int i = 0; i < HalfBlock; i++)
            {
                long elemIdx0 = baseIdx + i;
                long elemIdx1 = baseIdx + i + HalfBlock;
                int qLow = (q[i] & 0x0F) - 8;
                int qHigh = ((q[i] >> 4) & 0x0F) - 8;
                if (elemIdx0 < elementCount) dst[elemIdx0] = scale * qLow;
                if (elemIdx1 < elementCount) dst[elemIdx1] = scale * qHigh;
            }
        }
    }
}
