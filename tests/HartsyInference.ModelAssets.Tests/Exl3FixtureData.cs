using System.Runtime.InteropServices;
using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>The committed 256x384 EXL3 fixture (see Fixtures/Exl3/README.md) and helpers that wrap it, or a variation of it, as a recipe.</summary>
internal sealed class Exl3FixtureData : IDisposable
{
    public const int InDim = 256, OutDim = 384;

    public byte[] Trellis { get; } = Read("trellis.i16");
    public byte[] WHat { get; } = Read("w_hat.f16");
    public byte[] WFused { get; } = Read("w_fused.f16");
    public Tensor Suh { get; } = Bytes(Read("suh.f16"), DType.F16, InDim);
    public Tensor Svh { get; } = Bytes(Read("svh.f16"), DType.F16, OutDim);
    public Tensor Mcg { get; } = McgTensor(unchecked((int)Exl3Format.McgMultiplier));

    public static byte[] Read(string name) => File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", "Exl3", name));

    public static Tensor Bytes(byte[] data, DType dtype, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), dtype);
        data.CopyTo(t.AsSpan<byte>());
        return t;
    }

    public static Tensor McgTensor(int value)
    {
        Tensor t = new(new TensorShape(1), DType.I32);
        t.AsSpan<int>()[0] = value;
        return t;
    }

    public QuantRecipe Recipe() => new()
    {
        Encoding = QuantEncoding.Exl3Trellis, Geometry = new BlockGeometry(16, 16), ScaleDType = DType.F16,
        LogicalRows = OutDim, LogicalCols = InDim, Exl3 = new Exl3Companions(Suh, Svh, Mcg, Exl3Format.SupportedBits),
    };

    public static Half[] Halves(byte[] data)
    {
        Half[] h = new Half[data.Length / 2];
        data.CopyTo(MemoryMarshal.AsBytes(h.AsSpan()));
        return h;
    }

    public void Dispose()
    {
        Suh.Dispose();
        Svh.Dispose();
        Mcg.Dispose();
    }
}
