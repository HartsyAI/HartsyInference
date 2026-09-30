using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using static HartsyInference.Cpu.Tests.Attention.AttentionFixtures;

namespace HartsyInference.Cpu.Tests.Attention;

public sealed class SparseLatentAttentionReferenceTests
{
    private static (float[] Out, float[] Expected) RunFixture()
    {
        System.Text.Json.JsonElement f = Load("sparse_latent_attention.json");
        int dim = f.GetProperty("dim").GetInt32(), slots = f.GetProperty("windowSlots").GetInt32();
        int mainRows = f.GetProperty("mainRows").GetInt32(), t = f.GetProperty("tokens").GetInt32();
        int h = f.GetProperty("heads").GetInt32(), k = f.GetProperty("k").GetInt32();
        System.Text.Json.JsonElement w = f.GetProperty("window"), m = f.GetProperty("main");
        LatentSource window = Source(LatentEncoding.Fp8E4M3Ue8m0x32, Bytes(w.GetProperty("codes")), Bytes(w.GetProperty("scales")), slots, dim);
        LatentSource main = Source(LatentEncoding.Fp4E2M1E4M3x16, Bytes(m.GetProperty("codes")), Bytes(m.GetProperty("scales")), mainRows, dim);
        try
        {
            using Tensor q = F32(Floats(f.GetProperty("query")), t, h, dim);
            using Tensor sink = F32(Floats(f.GetProperty("sink")), h);
            using Tensor idx = I32(Ints(f.GetProperty("indices")), t, k);
            using Tensor o = Empty(DType.F32, t, h, dim);
            using CpuBackend cpu = new();
            cpu.SparseLatentAttention(o, q, window, main, idx, slots, sink, (float)f.GetProperty("scale").GetDouble());
            return (ReadF32(o), Floats(f.GetProperty("expected")));
        }
        finally { Dispose(window); Dispose(main); }
    }

    [Fact]
    public void Matches_The_Float64_Torch_Reference_With_Sink_And_Skipped_Indices()
    {
        (float[] actual, float[] expected) = RunFixture();
        Assert.True(MaxAbsDiff(expected, actual) < 1e-5f, $"max diff {MaxAbsDiff(expected, actual)}");
    }

    [Fact]
    public void A_Token_With_No_Valid_Index_Yields_Zeros()
    {
        (float[] actual, _) = RunFixture();
        const int Heads = 3, Dim = 64;
        Assert.All(actual[..(Heads * Dim)], v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Indices_Below_The_Window_Size_Read_The_Ring_And_The_Rest_Read_Main()
    {
        const int Dim = 32;
        float[] ring = Enumerable.Range(0, 2 * Dim).Select(i => i < Dim ? 1f : 2f).ToArray();
        float[] mainRows = Enumerable.Repeat(4f, Dim).ToArray();
        using Tensor rt = F32(ring, 2, Dim), mt = F32(mainRows, 1, Dim);
        LatentSource window = new(LatentEncoding.F32, rt, null, 2, Dim), main = new(LatentEncoding.F32, mt, null, 1, Dim);
        using Tensor q = F32(new float[Dim], 1, 1, Dim);      // zero query: uniform weights over valid rows
        using Tensor sink = F32(new[] { float.NegativeInfinity }, 1);
        using Tensor idx = I32(new[] { 1, 2, -1 }, 1, 3);      // ring slot 1 (value 2) and main row 0 (value 4)
        using Tensor o = Empty(DType.F32, 1, 1, Dim);
        using CpuBackend cpu = new();
        cpu.SparseLatentAttention(o, q, window, main, idx, 2, sink, 1f);
        Assert.All(ReadF32(o), v => Assert.Equal(3f, v, 5));
    }

    [Fact]
    public void The_Sink_Only_Enlarges_The_Denominator()
    {
        const int Dim = 32;
        using Tensor rt = F32(Enumerable.Repeat(2f, Dim).ToArray(), 1, Dim);
        LatentSource window = new(LatentEncoding.F32, rt, null, 1, Dim);
        using Tensor q = F32(new float[Dim], 1, 1, Dim);
        using Tensor sink = F32(new[] { 0f }, 1);              // exp(0) joins one valid exp(0) => half the value
        using Tensor idx = I32(new[] { 0 }, 1, 1);
        using Tensor o = Empty(DType.F32, 1, 1, Dim);
        using CpuBackend cpu = new();
        cpu.SparseLatentAttention(o, q, window, LatentSource.Empty, idx, 1, sink, 1f);
        Assert.All(ReadF32(o), v => Assert.Equal(1f, v, 5));
    }

    [Fact]
    public void Invalid_Operands_Throw()
    {
        const int Dim = 32;
        using Tensor rt = F32(new float[2 * Dim], 2, Dim);
        LatentSource window = new(LatentEncoding.F32, rt, null, 2, Dim);
        using Tensor q = F32(new float[Dim], 1, 1, Dim), sink = F32(new float[1], 1), idx = I32(new[] { 0 }, 1, 1);
        using Tensor o = Empty(DType.F32, 1, 1, Dim);
        using CpuBackend cpu = new();
        Assert.Throws<ArgumentException>(() => cpu.SparseLatentAttention(o, q, window, LatentSource.Empty, idx, 3, sink, 1f));
        using Tensor badSink = F32(new float[2], 2);
        Assert.Throws<ArgumentException>(() => cpu.SparseLatentAttention(o, q, window, LatentSource.Empty, idx, 2, badSink, 1f));
    }
}
