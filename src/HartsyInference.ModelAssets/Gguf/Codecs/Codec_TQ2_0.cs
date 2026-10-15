using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>TQ2_0: ternary in 2-bit codes, 256 elements per block, 66 bytes. Layout: <c>[64 bytes qs][2 bytes FP16 scale]</c>. Value =
/// <c>(code − 1) · scale</c>; element order follows canonical ggml <c>dequantize_row_tq2_0</c>: for each 32-byte half of <c>qs</c>,
/// bit pair <c>l</c> of every byte forms a run of 32 values.</summary>
public sealed unsafe class Codec_TQ2_0 : GgufCodecBase
{
    public override DType DType => DType.TQ2_0;

    private const int BlockElems = 256;
    private const int BlockBytes = 66;
    private const int QsBytes = 64;

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long numBlocks = elementCount / BlockElems;
        for (long b = 0; b < numBlocks; b++)
        {
            byte* qs = src + b * BlockBytes;
            float d = GgufCodecHelpers.ReadHalf(qs + QsBytes);
            float* y = dst + b * BlockElems;
            for (int j = 0; j < QsBytes; j += 32)
                for (int l = 0; l < 4; l++)
                    for (int m = 0; m < 32; m++) *y++ = (((qs[j + m] >> (l * 2)) & 3) - 1) * d;
        }
    }
}
