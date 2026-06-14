using BenchmarkDotNet.Attributes;
using SharpInference.Core.Tensors;

namespace SharpInference.GpuBenchmarks;

/// <summary>Elementwise (Silu, Gelu) and broadcast-add benchmarks. Useful to characterize the
/// fixed-overhead floor of any pointwise op — kernel launch + memory access — separately from the
/// cost of the work being done.</summary>
[Config(typeof(GpuBenchmarkConfig))]
public class ElementwiseGpuBenchmarks
{
    private BenchmarkFixture? _fixture;
    private Tensor? _input, _output, _bias;

    [ParamsSource(nameof(SizeSource))]
    public int SizeIndex { get; set; }
    public IEnumerable<int> SizeSource => Enumerable.Range(0, _sizes.Length);

    /// <summary>(B, C, H, W) — common diffusion activation shapes. Last entry is "small" to capture
    /// the launch-overhead floor.</summary>
    private static readonly (int B, int C, int H, int W)[] _sizes =
    [
        (1, 4096, 1, 1280),    // [B, S, H] flattened — DiT block residual
        (1, 1280, 32, 32),     // SDXL UNet bottom level
        (1, 320, 128, 128),    // SDXL UNet base level
        (1, 64, 1, 1),         // launch-overhead floor case
    ];

    [GlobalSetup]
    public void Setup()
    {
        _fixture = new BenchmarkFixture();
        (int B, int C, int H, int W) = _sizes[SizeIndex];
        _input = BenchmarkFixture.AllocateF32(new TensorShape(B, C, H, W));
        _output = new Tensor(new TensorShape(B, C, H, W), DType.F32);
        _bias = BenchmarkFixture.AllocateF32(new TensorShape(C));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _input?.Dispose(); _output?.Dispose(); _bias?.Dispose();
        _fixture?.Dispose();
    }

    [Benchmark]
    public void Silu()
    {
        _fixture!.Backend.Silu(_output!, _input!);
        _fixture.Sync();
    }

    [Benchmark]
    public void Gelu()
    {
        _fixture!.Backend.Gelu(_output!, _input!);
        _fixture.Sync();
    }

    /// <summary>BroadcastAdd of <c>[C]</c> bias into <c>[B, C, H, W]</c> — the AdaLN / per-channel-bias
    /// pattern. Stresses the broadcast logic, which is currently a separate kernel.</summary>
    [Benchmark]
    public void BroadcastAdd()
    {
        (int _, int C, int H, int W) = _sizes[SizeIndex];
        _fixture!.Backend.BroadcastAdd(_input!, _bias!, C, H * W);
        _fixture.Sync();
    }
}
