using SharpInference.Core.Tensors;

namespace SharpInference.ModelHandler.Gguf.Codecs;

/// <summary>IQ4_NL (4-bit non-linear i-quant): 32 elements / 18 bytes. Layout: <c>[2 bytes FP16 scale][16 bytes nibbles]</c> — same on-disk shape as Q4_0, but the 4-bit values are looked up in a 16-entry codepoint table (non-linear quantization). The table is fixed (canonical ggml) — it captures the typical distribution of weight values better than a uniform Q4_0.
///
/// <para>Reconstruction: <c>x[i] = scale * kvalues_iq4nl[q[i]]</c> where <c>kvalues_iq4nl</c> is the 16-entry signed codepoint table.</para>
///
/// <para>The codepoint table comes from llama.cpp's `kvalues_iq4nl` constant. Same nibble-packing as Q4_0 (low nibbles for elements 0..15, high nibbles for 16..31).</para></summary>
public sealed unsafe class Codec_IQ4_NL : GgufCodecBase
{
    public override DType DType => DType.IQ4_NL;

    private const int BlockElems = 32;
    private const int BlockBytes = 18;
    private const int HalfBlock = 16;

    /// <summary>Canonical ggml `kvalues_iq4nl` codepoint table — 16 signed integer values that act as the dequantization lookup. Verbatim from `ggml-quants.c`.</summary>
    private static readonly sbyte[] KValues = new sbyte[]
    {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    };

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long numBlocks = (elementCount + BlockElems - 1) / BlockElems;
        fixed (sbyte* lookup = KValues)
        {
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
                    int qLow = q[i] & 0x0F;
                    int qHigh = (q[i] >> 4) & 0x0F;
                    if (elemIdx0 < elementCount) dst[elemIdx0] = scale * lookup[qLow];
                    if (elemIdx1 < elementCount) dst[elemIdx1] = scale * lookup[qHigh];
                }
            }
        }
    }
}
