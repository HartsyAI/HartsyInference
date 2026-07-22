using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Gguf.Codecs;

/// <summary>MXFP4: OCP MX microscaling 4-bit, 32 elements per block, 17 bytes. Layout: <c>[1 byte E8M0 scale][16 bytes E2M1 nibbles]</c>. The first 16 elements use low nibbles, the last 16 use high nibbles (canonical ggml `dequantize_row_mxfp4`). Reconstruction: <c>x[i] = kvalues_mxfp4[q[i]] * 2^(e-128)</c> — the codeword table (<see cref="Kvalues"/>) is 2x the true E2M1 magnitudes, so the E8M0 scale is halved (exponent bias 128, not the usual 127) to compensate.</summary>
public sealed unsafe class Codec_MXFP4 : GgufCodecBase
{
    public override DType DType => DType.MXFP4;

    private const int BlockElems = 32;
    private const int BlockBytes = 17;
    private const int HalfBlock = 16;

    // ggml's kvalues_mxfp4 (= kvalues_fp4): signed E2M1 codewords, doubled to keep the table integral.
    private static readonly sbyte[] Kvalues = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

    public override void DequantizeToF32(byte* src, float* dst, long elementCount)
    {
        long numBlocks = (elementCount + BlockElems - 1) / BlockElems;
        for (long b = 0; b < numBlocks; b++)
        {
            byte* block = src + b * BlockBytes;
            float scale = E8M0ToFp32Half(block[0]);
            byte* q = block + 1;
            long baseIdx = b * BlockElems;
            for (int i = 0; i < HalfBlock; i++)
            {
                long elemIdx0 = baseIdx + i;
                long elemIdx1 = baseIdx + i + HalfBlock;
                float x0 = Kvalues[q[i] & 0x0F] * scale;
                float x1 = Kvalues[(q[i] >> 4) & 0x0F] * scale;
                if (elemIdx0 < elementCount) dst[elemIdx0] = x0;
                if (elemIdx1 < elementCount) dst[elemIdx1] = x1;
            }
        }
    }

    /// <summary>Canonical ggml `ggml_e8m0_to_fp32_half`: <c>2^(x-128)</c> (half of the standard E8M0→FP32 conversion, to match <see cref="Kvalues"/> being 2x the true E2M1 values). NaN (x=255) is not special-cased — matches upstream ggml, real MXFP4 checkpoints don't emit it.</summary>
    private static float E8M0ToFp32Half(byte x)
    {
        uint bits = x < 2 ? (uint)0x00200000 << x : (uint)(x - 1) << 23;
        return BitConverter.UInt32BitsToSingle(bits);
    }
}
