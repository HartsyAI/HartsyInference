using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Builds latent sources by quantizing random rows on the CPU reference, plus small byte helpers.</summary>
internal static unsafe class LatentVulkanTestData
{
    /// <summary>A source of <paramref name="rows"/> random rows, quantized by the CPU reference; caller disposes.</summary>
    public static LatentSource MakeSource(LatentEncoding enc, int rows, int dim, int seed, float amp = 3f)
    {
        if (rows == 0) return LatentSource.Empty;
        LatentSource s = Empty(enc, rows, dim);
        using Tensor src = Dsv41TestData.F32(Dsv41TestData.Random(rows * dim, seed, amp), rows, dim);
        using Tensor phys = Dsv41TestData.I32(Enumerable.Range(0, rows).ToArray(), rows);
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

    public static Tensor U8(byte[] data, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.U8);
        data.AsSpan().CopyTo(new Span<byte>((byte*)t.DataPointer, data.Length));
        return t;
    }
}
