using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>Q2_0: 2-bit, 64 elements per block, 18 bytes. Layout: <c>[2 bytes FP16 scale][16 bytes codes]</c>. Element <c>j</c> is the
/// code at bit <c>2·(j % 4)</c> of byte <c>j / 4</c>, and its value is <c>(code − 1) · scale</c>, the grid <c>{−1, 0, 1, 2}</c> (canonical
/// ggml <c>dequantize_row_q2_0</c>; the expert-pack <c>Q2_0Codec</c> writes the same blocks).</summary>
public sealed unsafe class Codec_Q2_0 : GgufCodecBase
{
    public override DType DType => DType.Q2_0;

    private const int BlockElems = 64;
    private const int BlockBytes = 18;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long numBlocks = elementCount / BlockElems;
        for (long b = 0; b < numBlocks; b++)
        {
            byte* block = src + b * BlockBytes;
            float d = GgufCodecHelpers.ReadHalf(block);
            byte* qs = block + 2;
            float* y = dst + b * BlockElems;
            for (int j = 0; j < BlockElems; j++) y[j] = (((qs[j >> 2] >> ((j & 3) * 2)) & 3) - 1) * d;
        }
    }
}
