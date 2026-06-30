using BenchmarkDotNet.Attributes;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;

namespace HartsyInference.GpuBenchmarks;

/// <summary>cuBLAS GEMM benchmarks across the shape grid hit by SDXL / Flux / SD3.5 / Z-Image / Flux2.
/// Each (M, N, K) shape comes from a real model — see comments on <see cref="ShapeIndex"/>. Both F32
/// and F16 paths are measured because diffusion runs typically mix dtypes (FP8 weights cast to F16,
/// F32 activations, etc.).</summary>
[Config(typeof(GpuBenchmarkConfig))]
public class MatMulGpuBenchmarks
{
    private BenchmarkFixture? _fixture;
    private Tensor? _a, _b, _outF32, _outF16;
    private Tensor? _aF16, _bF16;
    // Linear-with-bias operands ([N,K] weight, [N] bias) for the epilogue-fusion A/B (HARTSY_EPILOGUE_FUSION).
    private Tensor? _wF16, _biasF16, _outLinF16;
    // FP8 operands ([M,K] input, [N,K] weight, F16 out) for the native-FP8 A/B (HARTSY_FP8_NATIVE, Ada only).
    private Tensor? _inF8, _wF8, _outF8Lin;

    /// <summary>Index into a curated shape grid. Each index maps to a (M, K, N) triple drawn from a
    /// real diffusion model's hot path. Intentionally not a Cartesian explosion — that would multiply
    /// runtime without improving the paper's coverage.</summary>
    [ParamsSource(nameof(ShapeSource))]
    public int ShapeIndex { get; set; }

    /// <summary>Returns the indices into the shape grid. BenchmarkDotNet enumerates all values.</summary>
    public IEnumerable<int> ShapeSource => Enumerable.Range(0, _shapes.Length);

    /// <summary>Curated GEMM shapes relevant to diffusion inference. Format: (M, K, N) with output
    /// <c>[M, N]</c>. Comments reference where the shape appears in our code.</summary>
    private static readonly (int M, int K, int N)[] _shapes =
    [
        // SDXL UNet attention QKV projections at 64×64 (4096 spatial tokens), 1280 channels
        (4096, 1280, 1280),
        // SDXL UNet linear-out at 64×64, 1280→1280
        (4096, 1280, 1280),
        // SDXL FeedForward GeGLU expand: 1280 → 2*5120 = 10240
        (4096, 1280, 10240),
        // SDXL FeedForward contract: 5120 → 1280
        (4096, 5120, 1280),
        // Flux DiT QKV at 1024 tokens, hidden=3072
        (1024, 3072, 9216),  // fused QKV: 3 * 3072
        // Flux DiT FFN at 1024 tokens
        (1024, 3072, 12288),
        // SD3.5 Medium Joint Block at 1024 tokens, hidden=1536
        (1024, 1536, 4608),
        // Z-Image at 1024 tokens, hidden=3840
        (1024, 3840, 11520),  // fused QKV: 3 * 3840
        // Lumina-Image-2.0 at 1024 tokens, hidden=2304
        (1024, 2304, 6912),  // fused QKV: 3 * 2304
        // Hunyuan Image 2.1 at 1024 tokens, hidden=3072
        (1024, 3072, 9216),
    ];

    /// <summary>Allocates input + output tensors for the current shape. Runs once per shape.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _fixture = new BenchmarkFixture();
        (int M, int K, int N) = _shapes[ShapeIndex];
        _a = BenchmarkFixture.AllocateF32(new TensorShape(M, K), seed: 1);
        _b = BenchmarkFixture.AllocateF32(new TensorShape(K, N), seed: 2);
        _outF32 = new Tensor(new TensorShape(M, N), DType.F32);
        _aF16 = BenchmarkFixture.AllocateF16(new TensorShape(M, K), seed: 1);
        _bF16 = BenchmarkFixture.AllocateF16(new TensorShape(K, N), seed: 2);
        _outF16 = new Tensor(new TensorShape(M, N), DType.F16);
        // Linear: out[M,N] = input[M,K] · weight[N,K]^T + bias[N]. Weight is row-major [outDim, inDim].
        _wF16 = BenchmarkFixture.AllocateF16(new TensorShape(N, K), seed: 3);
        _biasF16 = BenchmarkFixture.AllocateF16(new TensorShape(N), seed: 4);
        _outLinF16 = new Tensor(new TensorShape(M, N), DType.F16);
        // FP8 operands for the native-FP8 A/B. Both input and weight must be FP8 for the cublasLtMatmul
        // path to dispatch; output is F16. On Ampere this falls back to cast-then-F16 (same as flag off).
        _inF8 = BenchmarkFixture.AllocateF8E4M3(new TensorShape(M, K), seed: 1);
        _wF8 = BenchmarkFixture.AllocateF8E4M3(new TensorShape(N, K), seed: 3);
        _outF8Lin = new Tensor(new TensorShape(M, N), DType.F16);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a?.Dispose(); _b?.Dispose(); _outF32?.Dispose();
        _aF16?.Dispose(); _bF16?.Dispose(); _outF16?.Dispose();
        _wF16?.Dispose(); _biasF16?.Dispose(); _outLinF16?.Dispose();
        _inF8?.Dispose(); _wF8?.Dispose(); _outF8Lin?.Dispose();
        _fixture?.Dispose();
    }

    /// <summary>F32 GEMM via cuBLAS SGEMM.</summary>
    [Benchmark]
    public void MatMul_F32()
    {
        _fixture!.Backend.MatMul(_outF32!, _a!, _b!);
        _fixture.Sync();
    }

    /// <summary>F16 GEMM via cuBLAS GemmEx (Tensor Cores when available).</summary>
    [Benchmark]
    public void MatMul_F16()
    {
        _fixture!.Backend.MatMul(_outF16!, _aF16!, _bF16!);
        _fixture.Sync();
    }

    /// <summary>F16 Linear (GEMM + bias). With <c>HARTSY_EPILOGUE_FUSION=1</c> the bias folds into the
    /// cuBLASLt epilogue (one launch); without it, this is cublasGemmEx + a separate BiasAdd. This is the
    /// A/B probe for Stage 2 of the quant/GEMM perf plan.</summary>
    [Benchmark]
    public void Linear_F16_Bias()
    {
        _fixture!.Backend.Linear(_outLinF16!, _aF16!, _wF16!, _biasF16!);
        _fixture.Sync();
    }

    /// <summary>FP8 Linear (E4M3 input + weight → F16 out). With <c>HARTSY_FP8_NATIVE=1</c> on Ada+ this
    /// dispatches the native cublasLtMatmul FP8 tensor-core path; otherwise (or on Ampere) it casts the
    /// FP8 operands up to F16 and runs cublasGemmEx. The A/B probe for Stage 4 of the quant/GEMM perf plan.</summary>
    [Benchmark]
    public void Linear_FP8_Native()
    {
        _fixture!.Backend.Linear(_outF8Lin!, _inF8!, _wF8!, bias: null);
        _fixture.Sync();
    }
}
