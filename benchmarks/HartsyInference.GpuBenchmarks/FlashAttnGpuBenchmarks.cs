using BenchmarkDotNet.Attributes;
using HartsyInference.Core.Tensors;

namespace HartsyInference.GpuBenchmarks;

/// <summary>Decode-shaped benchmark for the LM online-softmax kernel <c>lm_flash_attn_f32</c>
/// (<see cref="HartsyInference.Cuda.CudaBackend.FlashAttention"/>) — the kernel Stage 5 of the quant/GEMM
/// perf plan optimizes. Distinct from <see cref="SdpaGpuBenchmarks"/>, which exercises the score-
/// materializing diffusion <c>ScaledDotProductAttention</c> path. Decode (Tq=1) loops over all keys
/// sequentially with a per-key reduction, so this is where the per-key barrier count dominates and where
/// the warp-shuffle (5a) / keys-parallel (5b) changes pay off. Sweeps KV length to show the scaling.</summary>
[Config(typeof(GpuBenchmarkConfig))]
public class FlashAttnGpuBenchmarks
{
    private BenchmarkFixture? _fixture;
    private Tensor? _q, _k, _v, _out;

    // Llama-3-ish head config: Hq=32, Hkv=8 (GQA group 4), D=128. Only KV length varies.
    private const int Hq = 32, Hkv = 8, Group = 4, D = 128;

    [ParamsSource(nameof(KvSource))]
    public int KvLen { get; set; }

    public IEnumerable<int> KvSource => new[] { 512, 2048, 8192 };

    [GlobalSetup]
    public void Setup()
    {
        _fixture = new BenchmarkFixture();
        // Decode: a single query position (Tq=1) attends to KvLen cached keys.
        _q = BenchmarkFixture.AllocateF32(new TensorShape(1, Hq, 1, D), seed: 1);
        _k = BenchmarkFixture.AllocateF32(new TensorShape(1, Hkv, KvLen, D), seed: 2);
        _v = BenchmarkFixture.AllocateF32(new TensorShape(1, Hkv, KvLen, D), seed: 3);
        _out = new Tensor(new TensorShape(1, Hq, 1, D), DType.F32);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _q?.Dispose(); _k?.Dispose(); _v?.Dispose(); _out?.Dispose();
        _fixture?.Dispose();
    }

    [Benchmark]
    public void FlashAttn_Decode_F32()
    {
        float scale = 1f / MathF.Sqrt(D);
        _fixture!.Backend.FlashAttention(_out!, _q!, _k!, _v!, kvLen: KvLen, kvGroup: Group,
            causal: true, qOffset: KvLen - 1, scale: scale);
        _fixture.Sync();
    }
}
