using System.Globalization;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.ThreeD.Tests.Parity;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Idol;

/// <summary>A gated rank-32 adjunct on each of Idol's four block matrices, CUDA against the CPU backend, at the shapes and row counts of the real model.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class IdolTurboCudaLinearTests
{
    private readonly ITestOutputHelper _output;

    public IdolTurboCudaLinearTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(3072, 1024, 4096, false)]
    [InlineData(1024, 1024, 4096, false)]
    [InlineData(4096, 1024, 4096, false)]
    [InlineData(1024, 4096, 4096, false)]
    [InlineData(3072, 1024, 4096, true)]
    [InlineData(4096, 1024, 12294, false)]
    [InlineData(1024, 4096, 12294, false)]
    [InlineData(4096, 1024, 4096, false, true)]
    [InlineData(1024, 1024, 12294, false, true)]
    public void AnAdjunctOnALinear_IsTheSameOnCudaAndOnTheCpu(int outDim, int inDim, int rows, bool f16Weight, bool secondOffTerm = false)
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaBackend cuda = new(0, PtxDir());
        CpuBackend cpu = new();
        using Tensor weight32 = IdolModelTests.Random(new TensorShape(outDim, inDim), 1, 0.05f);
        using Tensor weight16 = f16Weight ? weight32.CastTo(DType.F16) : weight32;
        using Tensor bias = IdolModelTests.Random(new TensorShape(outDim), 2, 0.1f);
        using Tensor down = IdolModelTests.Random(new TensorShape(32, inDim), 3, 0.1f);
        using Tensor up = IdolModelTests.Random(new TensorShape(outDim, 32), 4, 0.1f);
        using Tensor input = IdolModelTests.Random(new TensorShape(1, rows, inDim), 5, 1f);
        AdjunctGate gate = new(), other = new();
        using Tensor down2 = IdolModelTests.Random(new TensorShape(32, inDim), 13, 0.1f);
        using Tensor up2 = IdolModelTests.Random(new TensorShape(outDim, 32), 14, 0.1f);
        List<LowRankAdjunctTerm> terms = [new LowRankAdjunctTerm { Down = down, Up = up, Scale = 1f, Gate = gate }];
        if (secondOffTerm)
            terms.Add(new LowRankAdjunctTerm { Down = down2, Up = up2, Scale = 1f, Gate = other });
        LowRankAdjunct adjunct = new() { Terms = terms };
        using Tensor carried = weight16.WithLowRankAdjunct(adjunct);

        using IDisposable scope = gate.Enable(1f);
        // The truth: the delta merged into a plain F32 weight, on the CPU.
        using Tensor merged = new(weight32.Shape, DType.F32);
        {
            ReadOnlySpan<float> wS = weight32.AsReadOnlySpan<float>(), dS = down.AsReadOnlySpan<float>(), uS = up.AsReadOnlySpan<float>();
            Span<float> m = merged.AsSpan<float>();
            for (int r = 0; r < outDim; r++)
                for (int k = 0; k < inDim; k++)
                {
                    float acc = wS[r * inDim + k];
                    for (int j = 0; j < 32; j++)
                        acc += uS[r * 32 + j] * dS[j * inDim + k];
                    m[r * inDim + k] = acc;
                }
        }

        using Tensor expected = new(new TensorShape(1, rows, outDim), DType.F32);
        cpu.Linear(expected, input, merged, bias);
        using Tensor actual = new(new TensorShape(1, rows, outDim), DType.F32);
        cuda.Linear(actual, input, carried, bias);
        cuda.Sync();

        (double rel, double maxAbs) = ParityCompare.RelativeL2(actual.AsReadOnlySpan<float>(), expected.AsReadOnlySpan<float>());
        _output.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{outDim}x{inDim} rows {rows} {(f16Weight ? "F16" : "F32")} weight: relRMS {rel:E2}, max abs {maxAbs:E2}"));
        Assert.True(rel < (f16Weight ? 2e-3 : 5e-4), $"relRMS {rel:E2}");
    }

    private static string PtxDir()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate))
                return candidate;
        }

        throw new DirectoryNotFoundException("src/HartsyInference.Cuda/Ptx not found above the test output directory.");
    }
}
