using BenchmarkDotNet.Attributes;
using SharpInference.Core.Tensors;
using SharpInference.Cpu.Kernels;

namespace SharpInference.Benchmarks;

/// <summary>Benchmarks for GroupNorm kernel measuring per-group mean/variance and affine transform cost.</summary>
[MemoryDiagnoser]
[SimpleJob]
public sealed class GroupNormBenchmarks
{
    private Tensor _input = null!;
    private Tensor _weight = null!;
    private Tensor _bias = null!;
    private Tensor _output = null!;

    /// <summary>Allocates 1x256x32x32 tensors and 256-element affine parameters for 32-group normalization.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _input = new Tensor(new TensorShape(1, 256, 32, 32), DType.F32);
        _weight = new Tensor(new TensorShape(256), DType.F32);
        _bias = new Tensor(new TensorShape(256), DType.F32);
        _output = new Tensor(new TensorShape(1, 256, 32, 32), DType.F32);
        FillDeterministic(_input);
        FillDeterministic(_weight);
        FillDeterministic(_bias);
    }

    /// <summary>Disposes all allocated tensors after benchmarks complete.</summary>
    [GlobalCleanup]
    public void Cleanup()
    {
        _input.Dispose();
        _weight.Dispose();
        _bias.Dispose();
        _output.Dispose();
    }

    /// <summary>GroupNorm with 32 groups on 256-channel 32x32 input, typical for U-Net mid blocks.</summary>
    [Benchmark]
    public void GroupNorm_32groups_256ch()
    {
        NormKernels.GroupNorm(_output, _input, _weight, _bias, 32, 1e-5f);
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
