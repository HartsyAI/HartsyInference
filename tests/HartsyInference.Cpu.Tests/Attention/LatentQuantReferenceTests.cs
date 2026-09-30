using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using static HartsyInference.Cpu.Tests.Attention.AttentionFixtures;

namespace HartsyInference.Cpu.Tests.Attention;

public sealed class LatentQuantReferenceTests
{
    private static readonly (string Key, LatentEncoding Enc)[] Encodings =
    {
        ("fp8", LatentEncoding.Fp8E4M3Ue8m0x32),
        ("fp4e4m3", LatentEncoding.Fp4E2M1E4M3x16),
        ("fp4e8m0", LatentEncoding.Fp4E2M1E8M0x32),
    };

    public static IEnumerable<object[]> Cases() => Encodings.Select(e => new object[] { e.Key, e.Enc });

    private static void AssertBitEqual(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"element {i}: expected {expected[i]:R} got {actual[i]:R}");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Quantized_Codes_And_Scale_Bytes_Match_The_Torch_Port_Byte_For_Byte(string key, LatentEncoding enc)
    {
        System.Text.Json.JsonElement f = Load("latent_quant_bytes.json");
        int rows = f.GetProperty("rows").GetInt32(), dim = f.GetProperty("dim").GetInt32();
        System.Text.Json.JsonElement q = f.GetProperty(key);
        byte[] expCodes = Bytes(q.GetProperty("codes")), expScales = Bytes(q.GetProperty("scales"));
        using Tensor input = F32(Floats(f.GetProperty("input")), rows, dim);
        using Tensor phys = I32(Enumerable.Range(0, rows).ToArray(), rows);
        LatentSource dest = Source(enc, new byte[expCodes.Length], new byte[expScales.Length], rows, dim);
        try
        {
            using CpuBackend cpu = new();
            cpu.QuantizeLatentRows(dest, input, phys);
            Assert.Equal(expCodes, ReadU8(dest.Codes!));
            Assert.Equal(expScales, ReadU8(dest.Scales!));
        }
        finally { Dispose(dest); }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void ActQuantDequantInPlace_Is_Bit_Identical_To_The_Torch_Inplace_Port(string key, LatentEncoding enc)
    {
        System.Text.Json.JsonElement f = Load("latent_quant_bytes.json");
        int rows = f.GetProperty("rows").GetInt32(), dim = f.GetProperty("dim").GetInt32();
        using Tensor x = F32(Floats(f.GetProperty("input")), rows, dim);
        using CpuBackend cpu = new();
        cpu.ActQuantDequantInPlace(x, enc);
        AssertBitEqual(Floats(f.GetProperty(key).GetProperty("dequant")), ReadF32(x));
    }

    [Fact]
    public void Quantize_Routes_Rows_By_Physical_Index_Skips_Negatives_And_Lets_Later_Rows_Win()
    {
        const int Dim = 32;
        float[] data = new float[3 * Dim];
        for (int i = 0; i < data.Length; i++) data[i] = (i / Dim + 1) * (1 + (i % Dim) * 0.03f);
        using Tensor input = F32(data, 3, Dim);
        using Tensor phys = I32(new[] { 2, -1, 2 }, 3);
        LatentSource dest = Source(LatentEncoding.Fp8E4M3Ue8m0x32, new byte[4 * Dim], new byte[4], 4, Dim);
        LatentSource ref3 = Source(LatentEncoding.Fp8E4M3Ue8m0x32, new byte[Dim], new byte[1], 1, Dim);
        try
        {
            using CpuBackend cpu = new();
            cpu.QuantizeLatentRows(dest, input, phys);
            using Tensor last = F32(data[(2 * Dim)..], 1, Dim);
            using Tensor one = I32(new[] { 0 }, 1);
            cpu.QuantizeLatentRows(ref3, last, one);
            Assert.Equal(ReadU8(ref3.Codes!), ReadU8(dest.Codes!)[(2 * Dim)..(3 * Dim)]);
            Assert.All(ReadU8(dest.Codes!)[..(2 * Dim)], b => Assert.Equal(0, b));
            Assert.Equal(0, ReadU8(dest.Scales!)[3]);
        }
        finally { Dispose(dest); Dispose(ref3); }
    }

    [Fact]
    public void Ue8m0_Scale_Is_The_Power_Of_Two_Ceiling_Of_Amax_Over_448()
    {
        const int Dim = 32;
        (float amax, byte expected)[] cases = { (448f, 127), (448.5f, 128), (224f, 126), (1f, 119), (0.5f, 118) };
        foreach ((float amax, byte expected) in cases)
        {
            float[] v = new float[Dim];
            v[3] = -amax;
            using Tensor input = F32(v, 1, Dim);
            using Tensor phys = I32(new[] { 0 }, 1);
            LatentSource dest = Source(LatentEncoding.Fp8E4M3Ue8m0x32, new byte[Dim], new byte[1], 1, Dim);
            try
            {
                using CpuBackend cpu = new();
                cpu.QuantizeLatentRows(dest, input, phys);
                Assert.Equal(expected, ReadU8(dest.Scales!)[0]);
            }
            finally { Dispose(dest); }
        }
    }

    [Fact]
    public void Error_Paths_Are_Explicit()
    {
        using CpuBackend cpu = new();
        using Tensor x = F32(new float[30], 1, 30);
        Assert.Throws<ArgumentException>(() => cpu.ActQuantDequantInPlace(x, LatentEncoding.Fp8E4M3Ue8m0x32));
        using Tensor rows = F32(new float[32], 1, 32);
        using Tensor phys = I32(new[] { 5 }, 1);
        LatentSource dest = Source(LatentEncoding.Fp8E4M3Ue8m0x32, new byte[64], new byte[2], 2, 32);
        try { Assert.Throws<ArgumentOutOfRangeException>(() => cpu.QuantizeLatentRows(dest, rows, phys)); }
        finally { Dispose(dest); }
        using Tensor wrong = U8(new byte[32], 1, 32);
        Assert.Throws<NotSupportedException>(() => cpu.ActQuantDequantInPlace(wrong, LatentEncoding.F32));
    }
}
