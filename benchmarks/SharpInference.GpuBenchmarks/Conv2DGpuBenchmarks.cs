using BenchmarkDotNet.Attributes;
using SharpInference.Core.Tensors;

namespace SharpInference.GpuBenchmarks;

/// <summary>Conv2D benchmarks at shapes that appear in SDXL UNet and the Flux/SD3 VAE. Three
/// kernel-size families: 3×3 stride 1 (UNet ResBlock conv, the common case), 3×3 stride 2 (UNet
/// downsample), 1×1 stride 1 (UNet skip + several VAE layers). Padding chosen to preserve spatial
/// dimensions where the model does (3×3 → pad 1; 1×1 → pad 0).</summary>
[Config(typeof(GpuBenchmarkConfig))]
public class Conv2DGpuBenchmarks
{
    private BenchmarkFixture? _fixture;
    private Tensor? _input, _weight, _bias, _output;

    [ParamsSource(nameof(ShapeSource))]
    public int ShapeIndex { get; set; }

    public IEnumerable<int> ShapeSource => Enumerable.Range(0, _shapes.Length);

    /// <summary>(N, Cin, Cout, H, W, K, stride, pad). Output spatial dims derive from H/W and stride.</summary>
    private static readonly (int N, int Cin, int Cout, int H, int W, int K, int Stride, int Pad)[] _shapes =
    [
        // SDXL UNet base level @ 1024² generation — 128² latent, 320 channels, 3×3
        (1, 320, 320, 128, 128, 3, 1, 1),
        // SDXL UNet middle level — 64², 640 channels
        (1, 640, 640, 64, 64, 3, 1, 1),
        // SDXL UNet bottom level — 32², 1280 channels
        (1, 1280, 1280, 32, 32, 3, 1, 1),
        // SDXL UNet downsample 320 → 320 stride 2
        (1, 320, 320, 128, 128, 3, 2, 1),
        // SDXL UNet residual skip — 1×1 conv to match channels
        (1, 320, 640, 128, 128, 1, 1, 0),
        // VAE decoder level 0 — 128 ch @ 256², 3×3
        (1, 128, 128, 256, 256, 3, 1, 1),
        // VAE decoder level 1 — 256 ch @ 512², 3×3 (larger)
        (1, 256, 256, 512, 512, 3, 1, 1),
        // VAE final upsample → 3 channels
        (1, 128, 3, 1024, 1024, 3, 1, 1),
    ];

    [GlobalSetup]
    public void Setup()
    {
        _fixture = new BenchmarkFixture();
        (int N, int Cin, int Cout, int H, int W, int K, int Stride, int Pad) = _shapes[ShapeIndex];
        _input = BenchmarkFixture.AllocateF32(new TensorShape(N, Cin, H, W), seed: 1);
        _weight = BenchmarkFixture.AllocateF32(new TensorShape(Cout, Cin, K, K), seed: 2);
        _bias = BenchmarkFixture.AllocateF32(new TensorShape(Cout), seed: 3);
        int outH = (H + 2 * Pad - K) / Stride + 1;
        int outW = (W + 2 * Pad - K) / Stride + 1;
        _output = new Tensor(new TensorShape(N, Cout, outH, outW), DType.F32);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _input?.Dispose(); _weight?.Dispose(); _bias?.Dispose(); _output?.Dispose();
        _fixture?.Dispose();
    }

    /// <summary>F32 Conv2D — current implementation is im2col + cuBLAS SGEMM. Phase B4.2 will add a
    /// cuDNN Winograd path; this benchmark serves as the baseline against which that lands.</summary>
    [Benchmark]
    public void Conv2D_F32()
    {
        (int _, int _, int _, int _, int _, int _, int Stride, int Pad) = _shapes[ShapeIndex];
        _fixture!.Backend.Conv2D(_output!, _input!, _weight!, _bias, Stride, Stride, Pad, Pad);
        _fixture.Sync();
    }
}
