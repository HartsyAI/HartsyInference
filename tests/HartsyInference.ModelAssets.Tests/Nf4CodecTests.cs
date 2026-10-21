using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Nf4;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Dequant parity for bitsandbytes NF4. Validates the codebook mapping, high-nibble-first
/// packing, per-block absmax scaling, and double-quant absmax reconstruction.</summary>
public sealed unsafe class Nf4CodecTests
{
    private const float Tol = 1e-6f;

    [Fact]
    public void Dequantize_AllSixteenCodes_MatchCodebook()
    {
        // 16 elements, nibbles 0..15. Byte j = (2j << 4) | (2j+1) → high nibble is the even element.
        Tensor packed = U8(8);
        Span<byte> p = packed.AsSpan<byte>();
        for (int j = 0; j < 8; j++)
        {
            p[j] = (byte)(((2 * j) << 4) | (2 * j + 1));
        }
        Tensor absmax = F32(new[] { 1.0f }); // one block of 16, scale 1.0

        Tensor outT = Nf4Codec.Dequantize(packed, absmax, new TensorShape(16), blockSize: 16);
        ReadOnlySpan<float> o = outT.AsReadOnlySpan<float>();
        for (int i = 0; i < 16; i++)
        {
            Assert.InRange(o[i], Nf4Codec.Nf4Lut[i] - Tol, Nf4Codec.Nf4Lut[i] + Tol);
        }

        packed.Dispose();
        absmax.Dispose();
        outT.Dispose();
    }

    [Fact]
    public void ReconstructDoubleQuantAbsmax_MatchesFormula()
    {
        // absmax[b] = nestedCodebook[q[b]] * nestedAbsmax[b/256] + offset
        Tensor q = U8(2);
        Span<byte> qs = q.AsSpan<byte>();
        qs[0] = 5;
        qs[1] = 200;

        Tensor nestedCodebook = F32(256);
        Span<float> cb = nestedCodebook.AsSpan<float>();
        for (int i = 0; i < 256; i++) cb[i] = i * 0.01f;

        Tensor nestedAbsmax = F32(new[] { 3.0f });
        const float offset = 0.1f;

        Tensor absmax = Nf4Codec.ReconstructDoubleQuantAbsmax(q, nestedAbsmax, nestedCodebook, offset, nestedBlockSize: 256);
        ReadOnlySpan<float> a = absmax.AsReadOnlySpan<float>();

        Assert.InRange(a[0], 0.05f * 3f + 0.1f - Tol, 0.05f * 3f + 0.1f + Tol); // 0.25
        Assert.InRange(a[1], 2.00f * 3f + 0.1f - Tol, 2.00f * 3f + 0.1f + Tol); // 6.1

        q.Dispose();
        nestedCodebook.Dispose();
        nestedAbsmax.Dispose();
        absmax.Dispose();
    }

    [Fact]
    public void Dequantize_RejectsNonU8()
    {
        Tensor bad = F32(8);
        Tensor absmax = F32(new[] { 1.0f });
        Assert.Throws<ArgumentException>(() => Nf4Codec.Dequantize(bad, absmax, new TensorShape(16), 16));
        bad.Dispose();
        absmax.Dispose();
    }

    private static Tensor U8(int count) => new Tensor(new TensorShape(count), DType.U8);

    private static Tensor F32(int count) => new Tensor(new TensorShape(count), DType.F32);

    private static Tensor F32(float[] values)
    {
        Tensor t = new Tensor(new TensorShape(values.Length), DType.F32);
        values.AsSpan().CopyTo(t.AsSpan<float>());
        return t;
    }

    [Fact]
    public void Quantize_RoundTrip_WithinHalfCodebookGapPerBlock()
    {
        const int n = 1000;
        float[] src = new float[n];
        Random rng = new(7);
        for (int i = 0; i < n; i++) src[i] = (float)(rng.NextDouble() * 2 - 1) * (1 + i % 5);

        (byte[] packed, float[] absmax) = Nf4Codec.Quantize(src, blockSize: 64);
        using Tensor pk = U8(packed.Length);
        packed.AsSpan().CopyTo(pk.AsSpan<byte>());
        using Tensor am = F32(absmax);
        using Tensor outT = Nf4Codec.Dequantize(pk, am, new TensorShape(n), blockSize: 64);

        ReadOnlySpan<float> o = outT.AsReadOnlySpan<float>();
        for (int i = 0; i < n; i++)
        {
            float bound = absmax[i / 64] * Nf4Codec.MaxBlockError + Tol;
            Assert.True(MathF.Abs(o[i] - src[i]) <= bound, $"element {i}: {o[i]} vs {src[i]}, bound {bound}");
        }
    }

    [Fact]
    public void Quantize_RoundTrip_OddLengthPartialLastBlock()
    {
        const int n = 21, blockSize = 16;
        float[] src = new float[n];
        Random rng = new(3);
        for (int i = 0; i < n; i++) src[i] = (float)(rng.NextDouble() * 2 - 1) * (1 + i % 4);

        (byte[] packed, float[] absmax) = Nf4Codec.Quantize(src, blockSize);
        Assert.Equal((n + 1) / 2, packed.Length);
        Assert.Equal(2, absmax.Length);
        using Tensor pk = U8(packed.Length);
        packed.AsSpan().CopyTo(pk.AsSpan<byte>());
        using Tensor am = F32(absmax);
        using Tensor outT = Nf4Codec.Dequantize(pk, am, new TensorShape(n), blockSize);

        ReadOnlySpan<float> o = outT.AsReadOnlySpan<float>();
        for (int i = 0; i < n; i++)
            Assert.True(MathF.Abs(o[i] - src[i]) <= absmax[i / blockSize] * Nf4Codec.MaxBlockError + Tol, $"element {i}");
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void Quantize_RejectsNonFiniteValues(float bad)
    {
        float[] src = new float[32];
        src[5] = bad;
        Assert.Throws<ArgumentException>(() => Nf4Codec.Quantize(src, blockSize: 16));
    }

    [Fact]
    public void QuantizeToCompanions_RoundTripsThroughProductionLoader()
    {
        const int rows = 8, cols = 40, blockSize = 64;
        float[] src = new float[rows * cols];
        Random rng = new(21);
        for (int i = 0; i < src.Length; i++) src[i] = (float)(rng.NextDouble() * 2 - 1) * (1 + i % 6);
        string dir = Directory.CreateTempSubdirectory("nf4-companions-").FullName;
        string path = Path.Combine(dir, "nf4.safetensors");
        try
        {
            List<KeyValuePair<string, Tensor>> companions = Nf4Codec.QuantizeToCompanions("layer.weight", src, rows, cols, "float32", blockSize);
            SafeTensorsMerger.Write(path, companions);
            foreach (KeyValuePair<string, Tensor> c in companions) c.Value.Dispose();

            (List<KeyValuePair<string, Tensor>> tensors, List<SafeTensorsLoader> loaders) = SafeTensorsMerger.Open(path);
            try
            {
                Assert.Equal(4, tensors.Count);
                List<Tensor> allocated = [];
                Dictionary<string, Tensor> folded = Nf4CompanionFold.Apply(tensors.ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal), allocated);
                try
                {
                    Tensor weight = folded["layer.weight"];
                    Assert.Equal(DType.F32, weight.DType);
                    ReadOnlySpan<float> decoded = weight.AsReadOnlySpan<float>();
                    Assert.Equal(src.Length, decoded.Length);
                    (byte[] _, float[] absmax) = Nf4Codec.Quantize(src, blockSize);
                    for (int i = 0; i < src.Length; i++)
                        Assert.True(MathF.Abs(decoded[i] - src[i]) <= absmax[i / blockSize] * Nf4Codec.MaxBlockError + Tol, $"element {i}");
                }
                finally
                {
                    foreach (Tensor t in allocated) t.Dispose();
                }
            }
            finally
            {
                foreach (SafeTensorsLoader loader in loaders) loader.Dispose();
            }
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Quantize_ZeroBlock_DecodesToZero()
    {
        float[] src = new float[20];
        (byte[] packed, float[] absmax) = Nf4Codec.Quantize(src, blockSize: 16);
        Assert.Equal(new float[] { 0f, 0f }, absmax);
        using Tensor pk = U8(packed.Length);
        packed.AsSpan().CopyTo(pk.AsSpan<byte>());
        using Tensor am = F32(absmax);
        using Tensor outT = Nf4Codec.Dequantize(pk, am, new TensorShape(20), blockSize: 16);
        foreach (float v in outT.AsReadOnlySpan<float>()) Assert.Equal(0f, v);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Quantize_RealTensor_RoundTripsWithinBound()
    {
        string? path = Environment.GetEnvironmentVariable("HARTSY_QUANT_TEST_SAFETENSORS");
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return;

        using SafeTensorsLoader loader = new();
        loader.Load(path);
        SafeTensorDescriptor d = loader.Descriptors.Values.First(x => x.DType == DType.F32 && x.Shape.Rank == 2 && x.Shape.ElementCount >= 4096);
        using Tensor w = loader.GetTensor(d.Name);
        ReadOnlySpan<float> src = w.AsReadOnlySpan<float>();
        long n = w.Shape.ElementCount;
        (byte[] packed, float[] absmax) = Nf4Codec.Quantize(src, blockSize: 64);
        using Tensor pk = U8(packed.Length);
        packed.AsSpan().CopyTo(pk.AsSpan<byte>());
        using Tensor am = F32(absmax);
        using Tensor outT = Nf4Codec.Dequantize(pk, am, new TensorShape(n), blockSize: 64);

        ReadOnlySpan<float> o = outT.AsReadOnlySpan<float>();
        for (long i = 0; i < n; i++)
            Assert.True(MathF.Abs(o[(int)i] - src[(int)i]) <= absmax[i / 64] * Nf4Codec.MaxBlockError + Tol, $"{d.Name}[{i}]");
    }
}
