using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.Nf4;
using Xunit;

namespace HartsyInference.ModelHandler.Tests;

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
    public void Dequantize_PerBlockAbsmax_Scales()
    {
        // 128 elements, all nibble 15 (LUT=+1.0), two blocks of 64 with scales 2.0 and 0.5.
        Tensor packed = U8(64);
        Span<byte> p = packed.AsSpan<byte>();
        for (int i = 0; i < 64; i++) p[i] = 0xFF; // both nibbles = 15
        Tensor absmax = F32(new[] { 2.0f, 0.5f });

        Tensor outT = Nf4Codec.Dequantize(packed, absmax, new TensorShape(128), blockSize: 64);
        ReadOnlySpan<float> o = outT.AsReadOnlySpan<float>();
        for (int i = 0; i < 64; i++) Assert.InRange(o[i], 2.0f - Tol, 2.0f + Tol);
        for (int i = 64; i < 128; i++) Assert.InRange(o[i], 0.5f - Tol, 0.5f + Tol);

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
}
