using BenchmarkDotNet.Attributes;
using SharpInference.Core.Tensors;

namespace SharpInference.GpuBenchmarks;

/// <summary>GroupNorm benchmarks at UNet/VAE shapes (32 groups standard).</summary>
[Config(typeof(GpuBenchmarkConfig))]
public class GroupNormGpuBenchmarks
{
    private BenchmarkFixture? _fixture;
    private Tensor? _input, _weight, _bias, _output;

    [ParamsSource(nameof(ShapeSource))]
    public int ShapeIndex { get; set; }
    public IEnumerable<int> ShapeSource => Enumerable.Range(0, Shapes.Length);

    private static readonly (int N, int C, int H, int W, int Groups)[] Shapes =
    [
        (1, 320, 128, 128, 32),
        (1, 640, 64, 64, 32),
        (1, 1280, 32, 32, 32),
        (1, 128, 256, 256, 32),
        (1, 256, 512, 512, 32),
    ];

    [GlobalSetup]
    public void Setup()
    {
        _fixture = new BenchmarkFixture();
        (int N, int C, int H, int W, int _) = Shapes[ShapeIndex];
        _input = BenchmarkFixture.AllocateF32(new TensorShape(N, C, H, W));
        _weight = BenchmarkFixture.AllocateF32(new TensorShape(C));
        _bias = BenchmarkFixture.AllocateF32(new TensorShape(C));
        _output = new Tensor(new TensorShape(N, C, H, W), DType.F32);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _input?.Dispose(); _weight?.Dispose(); _bias?.Dispose(); _output?.Dispose();
        _fixture?.Dispose();
    }

    [Benchmark]
    public void GroupNorm()
    {
        int groups = Shapes[ShapeIndex].Groups;
        _fixture!.Backend.GroupNorm(_output!, _input!, _weight!, _bias!, groups, 1e-6f);
        _fixture.Sync();
    }

    /// <summary>Already-fused GroupNorm + SiLU. Phase B's only existing fusion. Used here as the
    /// benchmark target for "what does fusion buy us?" — see B4.3 for additional fused kernels.</summary>
    [Benchmark]
    public void GroupNormSilu_Fused()
    {
        int groups = Shapes[ShapeIndex].Groups;
        _fixture!.Backend.GroupNormSilu(_output!, _input!, _weight!, _bias!, groups, 1e-6f);
        _fixture.Sync();
    }
}

/// <summary>LayerNorm benchmarks at DiT block hidden dims.</summary>
[Config(typeof(GpuBenchmarkConfig))]
public class LayerNormGpuBenchmarks
{
    private BenchmarkFixture? _fixture;
    private Tensor? _input, _weight, _bias, _output;

    [ParamsSource(nameof(ShapeSource))]
    public int ShapeIndex { get; set; }
    public IEnumerable<int> ShapeSource => Enumerable.Range(0, Shapes.Length);

    /// <summary>(B, S, H) — DiT block sizes.</summary>
    private static readonly (int B, int S, int H)[] Shapes =
    [
        (1, 4096, 1280),  // SDXL spatial
        (1, 1024, 1536),  // SD3.5 Medium
        (1, 1024, 3072),  // Flux / Hunyuan / OmniGen2
        (1, 1024, 3840),  // Z-Image
        (1, 1024, 2304),  // Lumina2
    ];

    [GlobalSetup]
    public void Setup()
    {
        _fixture = new BenchmarkFixture();
        (int B, int S, int H) = Shapes[ShapeIndex];
        _input = BenchmarkFixture.AllocateF32(new TensorShape(B, S, H));
        _weight = BenchmarkFixture.AllocateF32(new TensorShape(H));
        _bias = BenchmarkFixture.AllocateF32(new TensorShape(H));
        _output = new Tensor(new TensorShape(B, S, H), DType.F32);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _input?.Dispose(); _weight?.Dispose(); _bias?.Dispose(); _output?.Dispose();
        _fixture?.Dispose();
    }

    [Benchmark]
    public void LayerNorm()
    {
        _fixture!.Backend.LayerNorm(_output!, _input!, _weight!, _bias!, 1e-6f);
        _fixture.Sync();
    }
}

/// <summary>RmsNorm benchmarks at DiT block hidden dims (no bias, simpler than LayerNorm).</summary>
[Config(typeof(GpuBenchmarkConfig))]
public class RmsNormGpuBenchmarks
{
    private BenchmarkFixture? _fixture;
    private Tensor? _input, _weight, _output;

    [ParamsSource(nameof(ShapeSource))]
    public int ShapeIndex { get; set; }
    public IEnumerable<int> ShapeSource => Enumerable.Range(0, Shapes.Length);

    private static readonly (int B, int S, int H)[] Shapes =
    [
        (1, 1024, 1536),  // SD3.5 Medium
        (1, 1024, 3072),  // Flux / Hunyuan / OmniGen2
        (1, 1024, 3840),  // Z-Image
        (1, 1024, 2304),  // Lumina2
        (1, 1024, 4096),  // ERNIE-Image
    ];

    [GlobalSetup]
    public void Setup()
    {
        _fixture = new BenchmarkFixture();
        (int B, int S, int H) = Shapes[ShapeIndex];
        _input = BenchmarkFixture.AllocateF32(new TensorShape(B, S, H));
        _weight = BenchmarkFixture.AllocateF32(new TensorShape(H));
        _output = new Tensor(new TensorShape(B, S, H), DType.F32);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _input?.Dispose(); _weight?.Dispose(); _output?.Dispose();
        _fixture?.Dispose();
    }

    [Benchmark]
    public void RmsNorm()
    {
        _fixture!.Backend.RmsNorm(_output!, _input!, _weight!, 1e-6f);
        _fixture.Sync();
    }
}
