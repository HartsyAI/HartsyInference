using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Core.Tests;

/// <summary><see cref="W4A8ConvRotCodec"/> against an independent transcription of comfy-kitchen's
/// <c>_pack_codes</c> / <c>_dequant_int4_grouped_to_int8</c> / <c>rotate_int8_convrot_weight</c> (@cdab4787a8). The
/// rotation is checked against an explicit product with the Hadamard matrix rather than the codec's butterfly.</summary>
public sealed class W4A8ConvRotCodecTests
{
    [Fact]
    public void FourBit_DecodesCodebookLevelsOnTheInt8Grid()
    {
        const int n = 2, k = 8, group = 4;
        int[,] codes = Codes(n, k, 16, seed: 1);
        float[] codebook = [-1f, -0.8f, -0.6f, -0.5f, -0.4f, -0.25f, -0.15f, -0.05f, 0.05f, 0.15f, 0.25f, 0.4f, 0.5f, 0.6f, 0.8f, 1f];
        float[] sRel = [100f, 60f, 127f, 10f];
        float[] sChannel = [0.5f, 0.25f];

        float[] decoded = Decode(Pack(codes, 4), sRel, sChannel, codebook, null, n, group, 0);

        Assert.Equal(Expected(codes, codebook, sRel, sChannel, null, n, k, group, bits: 4), decoded);
    }

    [Fact]
    public void SixBit_DecodesUniformCodesFromBothPlanes()
    {
        const int n = 3, k = 8, group = 4;
        int[,] codes = Codes(n, k, 64, seed: 2);
        float[] sRel = [2f, 3f, 1f, 4f, 1.5f, 2.5f];
        float[] sChannel = [0.5f, 0.125f, 1f];

        float[] decoded = Decode(Pack(codes, 6), sRel, sChannel, null, null, n, group, 0);

        Assert.Equal(Expected(codes, null, sRel, sChannel, null, n, k, group, bits: 6), decoded);
    }

    [Fact]
    public void Correction_AddsPerGroupTerms()
    {
        const int n = 2, k = 8, group = 4;
        int[,] codes = Codes(n, k, 16, seed: 3);
        float[] codebook = Enumerable.Range(0, 16).Select(i => (i - 7.5f) / 8f).ToArray();
        float[] sRel = [16f, 32f, 64f, 8f];
        float[] sChannel = [0.5f, 0.5f];
        float[] correction = [0.5f, -1f, 2f, 0.25f]; // [groups, n]

        float[] decoded = Decode(Pack(codes, 4), sRel, sChannel, codebook, correction, n, group, 0);

        Assert.Equal(Expected(codes, codebook, sRel, sChannel, correction, n, k, group, bits: 4), decoded);
    }

    [Fact]
    public void ConvRot_UnrotatesEachRowByTheHadamard()
    {
        const int n = 2, k = 16, group = 4, rot = 16;
        int[,] codes = Codes(n, k, 16, seed: 4);
        float[] codebook = Enumerable.Range(0, 16).Select(i => (i - 7.5f) / 8f).ToArray();
        float[] sRel = Enumerable.Range(0, n * k / group).Select(i => 40f + i).ToArray();
        float[] sChannel = [0.01f, 0.02f];

        float[] decoded = Decode(Pack(codes, 4), sRel, sChannel, codebook, null, n, group, rot);

        float[] rotated = Expected(codes, codebook, sRel, sChannel, null, n, k, group, bits: 4);
        float[] h = Int8ConvRotCodec.BuildHadamard(rot);
        for (int r = 0; r < n; r++)
        {
            for (int j = 0; j < k; j++)
            {
                double sum = 0;
                for (int i = 0; i < k; i++)
                    sum += rotated[r * k + i] * h[i * k + j];
                Assert.Equal((float)sum, decoded[r * k + j], 0.02f); // BF16 output
            }
        }
    }

    [Fact]
    public void Int8Grid_WithTheChannelScale_IsTheSameWeightAsTheDenseDecode()
    {
        const int n = 3, k = 16, group = 4, rot = 16;
        int[,] codes = Codes(n, k, 16, seed: 5);
        float[] codebook = Enumerable.Range(0, 16).Select(i => (i - 7.5f) / 8f).ToArray();
        float[] sRel = Enumerable.Range(0, n * k / group).Select(i => 90f + 3 * i).ToArray();
        float[] sChannel = [0.01f, 0.03f, 0.002f];
        using Tensor packed = new Tensor(new TensorShape(n, k / 2), DType.I8);
        Pack(codes, 4).CopyTo(packed.AsSpan<byte>());
        using Tensor rel = F32(sRel, n, k / group);
        using Tensor chan = F32(sChannel, n);
        using Tensor cb = F32(codebook, 16);

        using Tensor grid = W4A8ConvRotCodec.DecodeToInt8(packed, rel, cb, group);
        using Tensor viaInt8 = Int8ConvRotCodec.DequantToBf16(grid, chan, rot);
        using Tensor dense = W4A8ConvRotCodec.DequantToBf16(packed, rel, chan, cb, null, group, rot);

        Assert.Equal((DType.I8, new TensorShape(n, k)), (grid.DType, grid.Shape));
        Assert.Equal(dense.AsReadOnlySpan<ushort>().ToArray(), viaInt8.AsReadOnlySpan<ushort>().ToArray());
    }

    [Fact]
    public void Width_DecidesTheBitDepth()
    {
        Assert.Equal(4, W4A8ConvRotCodec.BitsFor(64, 128));
        Assert.Equal(6, W4A8ConvRotCodec.BitsFor(96, 128));
        Assert.Throws<ArgumentException>(() => W4A8ConvRotCodec.BitsFor(100, 128));
    }

    [Fact]
    public void SixBit_RefusesACodebook()
    {
        using Tensor packed = new Tensor(new TensorShape(1, 6), DType.I8);
        using Tensor sRel = F32([1f, 1f], 1, 2);
        using Tensor sChannel = F32([1f], 1);
        using Tensor codebook = F32(new float[16], 16);

        Assert.Throws<ArgumentException>(() => W4A8ConvRotCodec.DequantToBf16(packed, sRel, sChannel, codebook, null, 4, 0));
    }

    private static float[] Decode(byte[] packed, float[] sRel, float[] sChannel, float[]? codebook, float[]? correction,
        int n, int group, int rot)
    {
        using Tensor q = new Tensor(new TensorShape(n, packed.Length / n), DType.I8);
        packed.CopyTo(q.AsSpan<byte>());
        using Tensor rel = F32(sRel, n, sRel.Length / n);
        using Tensor chan = F32(sChannel, n);
        using Tensor? cb = codebook is null ? null : F32(codebook, 16);
        using Tensor? corr = correction is null ? null : F32(correction, correction.Length / n, n);
        using Tensor bf16 = W4A8ConvRotCodec.DequantToBf16(q, rel, chan, cb, corr, group, rot);
        Assert.Equal(DType.BF16, bf16.DType);
        using Tensor f32 = bf16.CastTo(DType.F32);
        return f32.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>comfy-kitchen's decode before rotation, in double then rounded through BF16 like the codec's output.</summary>
    private static float[] Expected(int[,] codes, float[]? codebook, float[] sRel, float[] sChannel, float[]? correction,
        int n, int k, int group, int bits)
    {
        int groups = k / group;
        float[] w = new float[n * k];
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < k; c++)
            {
                float level = codebook is not null ? codebook[codes[r, c]] : codes[r, c] - (1 << (bits - 1));
                float q = Math.Clamp(MathF.Round(level * sRel[r * groups + c / group], MidpointRounding.ToEven), -127f, 127f);
                float v = q * sChannel[r];
                if (correction is not null)
                    v += correction[(c / group) * n + r];
                w[r * k + c] = v;
            }
        }
        using Tensor f = F32(w, n, k);
        using Tensor bf = f.CastTo(DType.BF16);
        using Tensor back = bf.CastTo(DType.F32);
        return back.AsReadOnlySpan<float>().ToArray();
    }

    /// <summary>comfy-kitchen's <c>_pack_codes</c>: a K/2-byte nibble plane (even column low), then at 6 bits a K/4-byte
    /// plane of each code's top two bits.</summary>
    private static byte[] Pack(int[,] codes, int bits)
    {
        int n = codes.GetLength(0), k = codes.GetLength(1);
        int width = bits == 4 ? k / 2 : k / 2 + k / 4;
        byte[] packed = new byte[n * width];
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < k; c += 2)
                packed[r * width + c / 2] = (byte)((codes[r, c] & 0xF) | ((codes[r, c + 1] & 0xF) << 4));
            if (bits == 6)
            {
                for (int c = 0; c < k; c++)
                    packed[r * width + k / 2 + c / 4] |= (byte)(((codes[r, c] >> 4) & 0x3) << (2 * (c % 4)));
            }
        }
        return packed;
    }

    private static int[,] Codes(int n, int k, int levels, int seed)
    {
        Random random = new Random(seed);
        int[,] codes = new int[n, k];
        for (int r = 0; r < n; r++)
            for (int c = 0; c < k; c++)
                codes[r, c] = random.Next(levels);
        return codes;
    }

    private static Tensor F32(float[] values, params long[] dims)
    {
        Tensor t = new Tensor(new TensorShape(dims), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }
}
