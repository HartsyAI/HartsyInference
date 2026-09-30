using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;

namespace HartsyInference.Cuda.Tests.Attention;

/// <summary>Builds latent sources by quantizing random rows on the CPU reference, plus small byte helpers.</summary>
internal static unsafe class LatentGpuTestData
{
    public static readonly LatentEncoding[] Encodings =
    {
        LatentEncoding.F32, LatentEncoding.Fp8E4M3Ue8m0x32, LatentEncoding.Fp4E2M1E4M3x16, LatentEncoding.Fp4E2M1E8M0x32,
    };

    /// <summary>A source of <paramref name="rows"/> random rows, quantized by the CPU reference; caller disposes.</summary>
    public static LatentSource MakeSource(LatentEncoding enc, int rows, int dim, int seed, float amp = 3f)
    {
        if (rows == 0) return LatentSource.Empty;
        LatentSource s = Empty(enc, rows, dim);
        using Tensor src = MoePrimitiveTestData.F32(MoePrimitiveTestData.Random(rows * dim, seed, amp), rows, dim);
        using Tensor phys = MoePrimitiveTestData.I32(Enumerable.Range(0, rows).ToArray(), rows);
        new CpuBackend().QuantizeLatentRows(s, src, phys);
        return s;
    }

    /// <summary>A zeroed source of the given geometry; caller disposes.</summary>
    public static LatentSource Empty(LatentEncoding enc, int rows, int dim)
    {
        if (enc == LatentEncoding.F32) return new(enc, new Tensor(new TensorShape(rows, dim), DType.F32), null, rows, dim);
        return new(enc, new Tensor(new TensorShape(rows, LatentEncodings.CodeBytesPerRow(enc, dim)), DType.U8),
            new Tensor(new TensorShape(rows, LatentEncodings.ScaleBytesPerRow(enc, dim)), DType.U8), rows, dim);
    }

    public static void Dispose(in LatentSource s)
    {
        s.Codes?.Dispose();
        s.Scales?.Dispose();
    }

    public static byte[] ReadU8(Tensor t) => new ReadOnlySpan<byte>((byte*)t.DataPointer, (int)t.ElementCount).ToArray();

    public static float MaxAbsDiff(float[] a, float[] b)
    {
        Xunit.Assert.Equal(a.Length, b.Length);
        float m = 0f;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    public static void AssertBitEqual(float[] cpu, float[] cuda, string what)
    {
        Xunit.Assert.Equal(cpu.Length, cuda.Length);
        for (int i = 0; i < cpu.Length; i++)
            // Every NaN counts as equal: its sign and payload carry no information (overflowed FP4 scales produce them).
            Xunit.Assert.True((float.IsNaN(cpu[i]) && float.IsNaN(cuda[i])) ||
                BitConverter.SingleToInt32Bits(cpu[i]) == BitConverter.SingleToInt32Bits(cuda[i]),
                $"{what} element {i}: cpu {cpu[i]:R} cuda {cuda[i]:R}");
    }
}
