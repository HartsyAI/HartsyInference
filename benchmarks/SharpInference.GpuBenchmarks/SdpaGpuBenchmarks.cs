using BenchmarkDotNet.Attributes;
using SharpInference.Core.Tensors;

namespace SharpInference.GpuBenchmarks;

/// <summary>Scaled Dot Product Attention benchmarks at the shapes that appear in our diffusion models.
/// Self-attention shapes have <c>Sq == Skv</c>; cross-attention shapes have <c>Skv</c> = text length.
/// The current <c>CudaBackend.ScaledDotProductAttention</c> materializes the full
/// <c>S = Q @ K^T / √d</c> matrix in GPU memory — a 6 GB allocation at SDXL 64×64 self-attn for one
/// batch. Phase B4.1 replaces this with a tiled FlashAttention 2 PTX kernel; this benchmark is the
/// before/after target.</summary>
[Config(typeof(GpuBenchmarkConfig))]
public class SdpaGpuBenchmarks
{
    private BenchmarkFixture? _fixture;
    private Tensor? _q, _k, _v, _outF32, _qF16, _kF16, _vF16, _outF16;

    [ParamsSource(nameof(ShapeSource))]
    public int ShapeIndex { get; set; }
    public IEnumerable<int> ShapeSource => Enumerable.Range(0, _shapes.Length);

    /// <summary>(B, H, Sq, Skv, D) — heads-leading layout used by our SDPA op.</summary>
    private static readonly (int B, int H, int Sq, int Skv, int D)[] _shapes =
    [
        // SDXL self-attention 32×32 (8 heads, head_dim=160 in some SDXL variants; using 80 for the
        // standard SDXL UNet)
        (1, 16, 1024, 1024, 80),
        // SDXL self-attention 64×64
        (1, 16, 4096, 4096, 80),
        // SDXL cross-attention 64×64 — KV is CLIP context (77 tokens, padded to 77)
        (1, 16, 4096, 77, 80),
        // SD3.5 Medium joint-attn 32×32 + 333 text tokens (256 T5 + 77 CLIP) on 1024 image tokens
        (1, 24, 1024 + 333, 1024 + 333, 64),
        // Flux Dev joint-attn 32×32 + 256 text tokens
        (1, 24, 1024 + 256, 1024 + 256, 128),
        // Z-Image 32×32 + caption tokens
        (1, 30, 1024 + 64, 1024 + 64, 128),
        // Lumina2 attention
        (1, 24, 1024, 1024, 96),
        // Cross-attn with very small KV — exercises the asymmetric case Flash Attention handles well
        (1, 16, 4096, 32, 80),
    ];

    [GlobalSetup]
    public void Setup()
    {
        _fixture = new BenchmarkFixture();
        (int B, int H, int Sq, int Skv, int D) = _shapes[ShapeIndex];

        _q = BenchmarkFixture.AllocateF32(new TensorShape(B, H, Sq, D), seed: 1);
        _k = BenchmarkFixture.AllocateF32(new TensorShape(B, H, Skv, D), seed: 2);
        _v = BenchmarkFixture.AllocateF32(new TensorShape(B, H, Skv, D), seed: 3);
        _outF32 = new Tensor(new TensorShape(B, H, Sq, D), DType.F32);

        _qF16 = BenchmarkFixture.AllocateF16(new TensorShape(B, H, Sq, D), seed: 1);
        _kF16 = BenchmarkFixture.AllocateF16(new TensorShape(B, H, Skv, D), seed: 2);
        _vF16 = BenchmarkFixture.AllocateF16(new TensorShape(B, H, Skv, D), seed: 3);
        _outF16 = new Tensor(new TensorShape(B, H, Sq, D), DType.F16);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _q?.Dispose(); _k?.Dispose(); _v?.Dispose(); _outF32?.Dispose();
        _qF16?.Dispose(); _kF16?.Dispose(); _vF16?.Dispose(); _outF16?.Dispose();
        _fixture?.Dispose();
    }

    [Benchmark]
    public void Sdpa_F32()
    {
        int D = _shapes[ShapeIndex].D;
        float scale = 1.0f / MathF.Sqrt(D);
        _fixture!.Backend.ScaledDotProductAttention(_outF32!, _q!, _k!, _v!, mask: null, scale);
        _fixture.Sync();
    }

    [Benchmark]
    public void Sdpa_F16()
    {
        int D = _shapes[ShapeIndex].D;
        float scale = 1.0f / MathF.Sqrt(D);
        _fixture!.Backend.ScaledDotProductAttention(_outF16!, _qF16!, _kF16!, _vF16!, mask: null, scale);
        _fixture.Sync();
    }
}
