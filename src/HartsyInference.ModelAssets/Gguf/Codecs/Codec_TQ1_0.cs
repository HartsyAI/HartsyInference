using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>TQ1_0: ternary, 256 elements per block, 54 bytes. Layout: <c>[48 bytes qs][4 bytes qh][2 bytes FP16 scale]</c>. Each <c>qs</c>
/// byte packs five base-3 digits and each <c>qh</c> byte four; digit <c>n</c> of byte <c>b</c> is <c>((uint8)(b·3ⁿ) · 3) &gt;&gt; 8</c>, and
/// the value is <c>(digit − 1) · scale</c>. Element order follows canonical ggml <c>dequantize_row_tq1_0</c>: the first 32 <c>qs</c>
/// bytes give 5×32 values digit-major, the last 16 give 5×16, then <c>qh</c> gives 4×4.</summary>
public sealed unsafe class Codec_TQ1_0 : GgufCodecBase
{
    public override DType DType => DType.TQ1_0;

    private const int BlockElems = 256;
    private const int BlockBytes = 54;
    private const int QsBytes = 48;
    private const int QhBytes = 4;

    private static readonly byte[] Pow3 = [1, 3, 9, 27, 81, 243];

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long numBlocks = elementCount / BlockElems;
        for (long b = 0; b < numBlocks; b++)
        {
            byte* block = src + b * BlockBytes;
            byte* qs = block;
            byte* qh = block + QsBytes;
            float d = GgufCodecHelpers.ReadHalf(block + QsBytes + QhBytes);
            float* y = dst + b * BlockElems;
            const int fullBytes = QsBytes - QsBytes % 32;
            for (int j = 0; j < fullBytes; j += 32)
                for (int n = 0; n < 5; n++)
                    for (int m = 0; m < 32; m++) *y++ = Trit(qs[j + m], n) * d;
            for (int j = fullBytes; j < QsBytes; j += 16)
                for (int n = 0; n < 5; n++)
                    for (int m = 0; m < 16; m++) *y++ = Trit(qs[j + m], n) * d;
            for (int n = 0; n < 4; n++)
                for (int j = 0; j < QhBytes; j++) *y++ = Trit(qh[j], n) * d;
        }
    }

    /// <summary>Digit <paramref name="n"/> of a base-3 packed byte, minus one.</summary>
    private static int Trit(byte packed, int n)
    {
        byte q = (byte)(packed * Pow3[n]);
        return ((q * 3) >> 8) - 1;
    }
}
