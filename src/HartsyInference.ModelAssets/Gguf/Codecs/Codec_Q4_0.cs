using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>Q4_0: 4-bit, 32 elements per block, 18 bytes. Layout: <c>[2 bytes FP16 scale][16 bytes nibbles]</c>. Reconstruction: <c>x[i] = scale * (q[i] - 8)</c> where <c>q[i] ∈ [0..15]</c>. The first 16 elements use low nibbles, the last 16 use high nibbles (canonical ggml `dequantize_row_q4_0`).</summary>
public sealed unsafe class Codec_Q4_0 : GgufCodecBase
{
    public override DType DType => DType.Q4_0;

    public override bool SupportsQuantize => true;

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

    /// <summary>ggml's <c>quantize_row_q4_0_ref</c>: the block's largest-magnitude value, sign kept, maps to −8, so
    /// <c>d = max / −8</c> and <c>q = min(15, (int)(x / d + 8.5))</c>.</summary>
    public override void QuantizeFromF32(float* src, byte* dst, long elementCount)
    {
        long numBlocks = (elementCount + BlockElems - 1) / BlockElems;
        for (long b = 0; b < numBlocks; b++)
        {
            long baseIdx = b * BlockElems;
            float absMax = 0f, max = 0f;
            for (int i = 0; i < BlockElems && baseIdx + i < elementCount; i++)
            {
                float v = src[baseIdx + i];
                if (MathF.Abs(v) > absMax)
                {
                    absMax = MathF.Abs(v);
                    max = v;
                }
            }
            float d = max / -8f;
            float id = d != 0f ? 1f / d : 0f;
            byte* block = dst + b * BlockBytes;
            *(Half*)block = (Half)d;
            byte* q = block + 2;
            for (int i = 0; i < HalfBlock; i++)
            {
                float x0 = baseIdx + i < elementCount ? src[baseIdx + i] * id : 0f;
                float x1 = baseIdx + i + HalfBlock < elementCount ? src[baseIdx + i + HalfBlock] * id : 0f;
                int q0 = Math.Min(15, (int)(x0 + 8.5f));
                int q1 = Math.Min(15, (int)(x1 + 8.5f));
                q[i] = (byte)(q0 | (q1 << 4));
            }
        }
    }
}
