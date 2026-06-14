using BenchmarkDotNet.Attributes;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu.Kernels;

namespace HartsyInference.Benchmarks;

/// <summary>Benchmarks for square matrix multiplication at varying dimensions.</summary>
[MemoryDiagnoser]
[SimpleJob]
public sealed class MatMulBenchmarks
{
    /// <summary>Matrix dimension N for [N,N] @ [N,N] multiply.</summary>
    [Params(64, 256, 1024)]
    public int N { get; set; }

    private Tensor _matA = null!;
    private Tensor _matB = null!;
    private Tensor _matOut = null!;

    /// <summary>Allocates square matrices with deterministic data before each parameter combination.</summary>
    [GlobalSetup]
    public void Setup()
    {
        TensorShape shape = new TensorShape(N, N);
        _matA = new Tensor(shape, DType.F32);
        _matB = new Tensor(shape, DType.F32);
        _matOut = new Tensor(shape, DType.F32);
        FillDeterministic(_matA);
        FillDeterministic(_matB);
    }

    /// <summary>Disposes all allocated tensors after benchmarks complete.</summary>
    [GlobalCleanup]
    public void Cleanup()
    {
        _matA.Dispose();
        _matB.Dispose();
        _matOut.Dispose();
    }

    /// <summary>Square matrix multiply [N,N] @ [N,N] testing SIMD and cache tiling effectiveness.</summary>
    [Benchmark]
    public void MatMul_NxN()
    {
        MatMulKernels.MatMul(_matOut, _matA, _matB);
    }

    private static void FillDeterministic(Tensor tensor)
    {
        Span<float> span = tensor.AsSpan<float>();
        for (int i = 0; i < span.Length; i++)
        {
            span[i] = i * 0.001f;
        }
    }
}
