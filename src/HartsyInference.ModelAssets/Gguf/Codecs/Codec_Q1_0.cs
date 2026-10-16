using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>Q1_0: 1-bit, 128 elements per block, 18 bytes. Layout: <c>[2 bytes FP16 scale][16 bytes sign bits]</c>. Element <c>j</c> is bit
/// <c>j % 8</c> of byte <c>j / 8</c>: <c>+scale</c> when set, <c>−scale</c> otherwise (canonical ggml <c>dequantize_row_q1_0</c>).</summary>
public sealed unsafe class Codec_Q1_0 : GgufCodecBase
{
    public override DType DType => DType.Q1_0;

    private const int BlockElems = 128;
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
            for (int j = 0; j < BlockElems; j++) y[j] = ((qs[j >> 3] >> (j & 7)) & 1) != 0 ? d : -d;
        }
    }
}
