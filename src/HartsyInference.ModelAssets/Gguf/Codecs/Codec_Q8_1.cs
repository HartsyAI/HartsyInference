using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>Q8_1: 8-bit + cached row sum, 32 elements / 40 bytes. Layout: <c>[2 scale][2 sum-cache (FP16)][4 reserved/padding][32 int8]</c>. Used primarily as an activation quant during inference; rare for stored weights but supported here for completeness. Reconstruction: <c>x[i] = scale * q[i]</c>; the sum field is ignored on dequant.
///
/// <para><b>Note on layout</b>: ggml's `block_q8_1` is <c>{half d; half s; int8_t qs[32]}</c> — 36 bytes by struct, but ggml pads to 40 bytes alignment. We honor the 40-byte stride so file offsets match.</para></summary>
public sealed unsafe class Codec_Q8_1 : GgufCodecBase
{
    public override DType DType => DType.Q8_1;

    private const int BlockElems = 32;
    private const int BlockBytes = 40;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long numBlocks = (elementCount + BlockElems - 1) / BlockElems;
        for (long b = 0; b < numBlocks; b++)
        {
            byte* block = src + b * BlockBytes;
            float scale = GgufCodecHelpers.ReadHalf(block);
            sbyte* q = (sbyte*)(block + 4);
            long baseIdx = b * BlockElems;
            for (int i = 0; i < BlockElems; i++)
            {
                long elemIdx = baseIdx + i;
                if (elemIdx >= elementCount) break;
                dst[elemIdx] = scale * q[i];
            }
        }
    }
}
