using BenchmarkDotNet.Attributes;
using SharpInference.Core.Tensors;
using SharpInference.Cpu.Kernels;

namespace SharpInference.Benchmarks;

/// <summary>Benchmarks for Conv2D kernels exercising im2col and GEMM-based convolution paths.</summary>
[MemoryDiagnoser]
[SimpleJob]
public sealed class Conv2DBenchmarks
{
    private Tensor _input = null!;
    private Tensor _weight = null!;
    private Tensor _output = null!;

    /// <summary>Allocates NCHW tensors for 3x3 convolution on 1x64x64x64 input.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _input = new Tensor(new TensorShape(1, 64, 64, 64), DType.F32);
        _weight = new Tensor(new TensorShape(64, 64, 3, 3), DType.F32);
        _output = new Tensor(new TensorShape(1, 64, 64, 64), DType.F32);
        FillDeterministic(_input);
        FillDeterministic(_weight);
    }

    /// <summary>Disposes all allocated tensors after benchmarks complete.</summary>
    [GlobalCleanup]
    public void Cleanup()
    {
        _input.Dispose();
        _weight.Dispose();
        _output.Dispose();
    }

    /// <summary>Conv2D with 3x3 kernel on 64-channel 64x64 input, stride=1, pad=1.</summary>
    [Benchmark]
    public void Conv2D_3x3_64ch_64x64()
    {
        Conv2DKernels.Conv2D(_output, _input, _weight, null, 1, 1, 1, 1);
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
