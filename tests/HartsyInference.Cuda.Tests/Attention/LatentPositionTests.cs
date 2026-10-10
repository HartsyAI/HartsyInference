using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;
using Xunit.Abstractions;
using static HartsyInference.Cuda.Tests.MoePrimitiveTestData;
using static HartsyInference.Cuda.Tests.Attention.LatentGpuTestData;

namespace HartsyInference.Cuda.Tests.Attention;

/// <summary>BuildWindowIndices and the offset interleaved RoPE on CUDA against the CPU reference; both must be exact.
/// Skips without CUDA.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed class LatentPositionTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(128, 5, 0)]
    [InlineData(128, 300, 0)]
    [InlineData(128, 1, 7)]
    [InlineData(128, 1, 128)]
    [InlineData(4, 1, 6)]
    public void WindowIndices_MatchCpuExactly(int window, int seqLen, int startPos)
    {
        if (!CudaContext.IsAvailable()) return;
        (int rows, int cols) = WindowIndicesReference.Shape(window, seqLen, startPos);

        int[] Run(IBackend be)
        {
            using Tensor idx = EmptyI32(rows, cols);
            be.BuildWindowIndices(idx, window, seqLen, startPos);
            return ReadI32(idx);
        }

        int[] cpu = Run(new CpuBackend());
        using CudaBackend cuda = new(0, PtxDir());
        Assert.Equal(cpu, Run(cuda));
        output.WriteLine($"window={window} seq={seqLen} start={startPos}: {rows}x{cols}, {cpu.Count(v => v < 0)} empty");
    }

    [Theory]
    [InlineData(3, 1, 512, 64, 448, false)]
    [InlineData(1, 7, 64, 64, 0, true)]
    public void RopeInterleavedOffset_IsBitIdenticalToCpu_ForwardAndInverse(int batch, int len, int dim, int rotary,
        int offset, bool withHeads)
    {
        if (!CudaContext.IsAvailable()) return;
        const int Heads = 6;
        float[] data = Random(batch * len * (withHeads ? Heads : 1) * dim, 91);
        float[] cos = Random(batch * len * (rotary / 2), 92), sin = Random(batch * len * (rotary / 2), 93);
        long[] shape = withHeads ? new long[] { batch, len, Heads, dim } : new long[] { batch, len, dim };

        (float[] Fwd, float[] Round) Run(IBackend be)
        {
            using Tensor x = F32(data, shape);
            using Tensor c = F32(cos, batch, len, rotary / 2), s = F32(sin, batch, len, rotary / 2);
            using Tensor negS = F32(sin.Select(v => -v).ToArray(), batch, len, rotary / 2);
            be.ApplyRopeInterleaved(x, c, s, rotary, offset);
            float[] fwd = ReadF32(x);
            be.ApplyRopeInterleaved(x, c, negS, rotary, offset);
            return (fwd, ReadF32(x));
        }

        (float[] Fwd, float[] Round) cpu = Run(new CpuBackend());
        using CudaBackend cuda = new(0, PtxDir());
        (float[] Fwd, float[] Round) gpu = Run(cuda);
        AssertBitEqual(cpu.Fwd, gpu.Fwd, "rope forward");
        AssertBitEqual(cpu.Round, gpu.Round, "rope inverse");
        for (int i = 0; i < data.Length; i++)                              // columns outside the slice never move
        {
            int col = i % dim;
            if (col < offset || col >= offset + rotary)
                Assert.Equal(data[i], gpu.Fwd[i]);
        }
    }

    [Fact]
    public void RopeInterleavedOffset_RejectsASliceOutsideTheVector()
    {
        if (!CudaContext.IsAvailable()) return;
        using CudaBackend cuda = new(0, PtxDir());
        using Tensor x = F32(new float[64], 1, 1, 64), c = F32(new float[32], 1, 1, 32), s = F32(new float[32], 1, 1, 32);
        Assert.Throws<ArgumentOutOfRangeException>(() => cuda.ApplyRopeInterleaved(x, c, s, 64, 2));
    }
}
