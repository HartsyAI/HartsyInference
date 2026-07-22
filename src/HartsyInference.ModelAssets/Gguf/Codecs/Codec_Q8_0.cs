using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>Q8_0: 8-bit quantization, 32 elements per block, 34 bytes/block. Layout: <c>[2 bytes FP16 scale][32 bytes int8 data]</c>. Reconstruction: <c>x[i] = scale * q[i]</c>.</summary>
public sealed unsafe class Codec_Q8_0 : GgufCodecBase
{
    public override DType DType => DType.Q8_0;
    public override bool SupportsQuantize => true;

    private const int BlockElems = 32;
    private const int BlockBytes = 34;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long numBlocks = (elementCount + BlockElems - 1) / BlockElems;
        for (long b = 0; b < numBlocks; b++)
        {
            byte* block = src + b * BlockBytes;
            float scale = GgufCodecHelpers.ReadHalf(block);
            sbyte* q = (sbyte*)(block + 2);
            long baseIdx = b * BlockElems;
            for (int i = 0; i < BlockElems; i++)
            {
                long elemIdx = baseIdx + i;
                if (elemIdx >= elementCount) break;
                dst[elemIdx] = scale * q[i];
            }
        }
    }

    public override void QuantizeFromF32(float* src, byte* dst, long elementCount)
    {
        long numBlocks = (elementCount + BlockElems - 1) / BlockElems;
        for (long b = 0; b < numBlocks; b++)
        {
            long baseIdx = b * BlockElems;
            float absMax = 0f;
            for (int i = 0; i < BlockElems; i++)
            {
                long elemIdx = baseIdx + i;
                if (elemIdx >= elementCount) break;
                float v = MathF.Abs(src[elemIdx]);
                if (v > absMax) absMax = v;
            }
            float scale = absMax / 127f;
            float invScale = scale > 0 ? 1f / scale : 0f;

            byte* block = dst + b * BlockBytes;
            *(Half*)block = (Half)scale;
            sbyte* q = (sbyte*)(block + 2);
            for (int i = 0; i < BlockElems; i++)
            {
                long elemIdx = baseIdx + i;
                if (elemIdx >= elementCount) { q[i] = 0; continue; }
                int rounded = (int)MathF.Round(src[elemIdx] * invScale);
                q[i] = (sbyte)Math.Clamp(rounded, -127, 127);
            }
        }
    }
}
