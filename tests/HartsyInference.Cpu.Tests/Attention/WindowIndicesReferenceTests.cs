using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using Xunit;
using static HartsyInference.Cpu.Tests.Attention.AttentionFixtures;

namespace HartsyInference.Cpu.Tests.Attention;

public sealed class WindowIndicesReferenceTests
{
    [Fact]
    public void Matches_Upstream_Get_Window_Topk_Idxs_For_Prefill_And_Decode()
    {
        using CpuBackend cpu = new();
        int count = 0;
        foreach (System.Text.Json.JsonElement c in Load("window_indices.json").GetProperty("cases").EnumerateArray())
        {
            int w = c.GetProperty("window").GetInt32(), s = c.GetProperty("seqLen").GetInt32(), p = c.GetProperty("startPos").GetInt32();
            (int rows, int cols) = WindowIndicesReference.Shape(w, s, p);
            Assert.Equal(c.GetProperty("rows").GetInt32(), rows);
            Assert.Equal(c.GetProperty("cols").GetInt32(), cols);
            using Tensor idx = Empty(DType.I32, rows, cols);
            cpu.BuildWindowIndices(idx, w, s, p);
            Assert.Equal(Ints(c.GetProperty("indices")), ReadI32(idx));
            count++;
        }
        Assert.Equal(11, count);
    }

    [Fact]
    public void Invalid_Operands_Throw()
    {
        using CpuBackend cpu = new();
        using Tensor idx = Empty(DType.I32, 4, 4);
        Assert.Throws<ArgumentOutOfRangeException>(() => cpu.BuildWindowIndices(idx, 0, 4, 0));
        Assert.Throws<ArgumentException>(() => cpu.BuildWindowIndices(idx, 4, 4, 2));
        using Tensor wrong = Empty(DType.I32, 3, 4);
        Assert.Throws<ArgumentException>(() => cpu.BuildWindowIndices(wrong, 4, 4, 0));
    }
}
