using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Ad-hoc measurement (not a correctness gate) for the Vulkan perf plan: drives a workload of
/// Flux-shape FP8 Linears so the <see cref="VulkanProfiler"/> (HARTSYINFERENCE_VK_PROFILE=1) dumps the
/// per-call time + dispatch count. Confirms on CURRENT code whether Linear is per-dispatch-overhead-bound
/// and how many dispatches a FP8 Linear costs (input cast + weight cast(s) + matmul + bias). Run with the
/// env set; read the dumped profile. Skips when Vulkan is unavailable.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanLinearProfileMeasurement
{
    private readonly ITestOutputHelper _out;
    public VulkanLinearProfileMeasurement(ITestOutputHelper output) => _out = output;

    [Fact]
    public unsafe void Measure_FluxFfn_FP8_Linear()
    {
        const int M = 1024, K = 3072, N = 12288;   // Flux DiT FFN expand, diffusion-scale M
        const int iters = 60;

        using VulkanBackend backend = new();

        Tensor input = new(new TensorShape(M, K), DType.F16);
        Tensor weight = new(new TensorShape(N, K), DType.F8E4M3);   // Flux Schnell ships FP8 weights
        Tensor bias = new(new TensorShape(N), DType.F16);
        Tensor output = new(new TensorShape(M, N), DType.F16);
        // Deterministic non-zero input so we don't hit any all-zero fast path.
        Half* ip = (Half*)input.DataPointer;
        for (long i = 0; i < (long)M * K; i++) ip[i] = (Half)(((i * 13) % 17 - 8) * 0.01f);

        backend.PreloadWeights(new[] { weight, bias });

        // Warm up (pipeline build, first upload), then the measured loop. The profiler accumulates over
        // all calls and dumps on backend Dispose; per-call = total / (iters + warmup).
        for (int i = 0; i < 5; i++) { backend.Linear(output, input, weight, bias); backend.Sync(); }
        for (int i = 0; i < iters; i++) { backend.Linear(output, input, weight, bias); backend.Sync(); }

        _out.WriteLine($"Ran {iters}+5 Flux-FFN FP8 Linears at M={M},K={K},N={N}. " +
            "Profile dump (if HARTSYINFERENCE_VK_PROFILE=1) on backend dispose shows dispatch count + per-call time.");

        input.Dispose(); weight.Dispose(); bias.Dispose(); output.Dispose();
        // backend.Dispose() (via using) triggers the profiler dump.
    }

    /// <summary>Stage-4 probe: many TINY Linears so per-call time is dominated by per-dispatch host overhead
    /// (descriptor alloc/update/bind + barrier + command record), not GEMM compute or upload. Reveals the
    /// floor the descriptor/barrier levers target. Run with HARTSYINFERENCE_VK_PROFILE=1.</summary>
    [Fact]
    public unsafe void Measure_TinyLinear_DispatchOverhead()
    {
        const int M = 16, K = 64, N = 64;   // coopmat-eligible, negligible compute
        const int iters = 2000;

        using VulkanBackend backend = new();

        Tensor weight = new(new TensorShape(N, K), DType.F16);
        Tensor bias = new(new TensorShape(N), DType.F16);
        Tensor input = new(new TensorShape(M, K), DType.F16);
        Tensor output = new(new TensorShape(M, N), DType.F16);
        Half* ip = (Half*)input.DataPointer;
        for (long i = 0; i < (long)M * K; i++) ip[i] = (Half)(((i * 13) % 17 - 8) * 0.01f);
        Half* wp = (Half*)weight.DataPointer;
        for (long i = 0; i < (long)N * K; i++) wp[i] = (Half)(((i * 7) % 13 - 6) * 0.01f);

        backend.PreloadWeights(new[] { weight, bias });

        for (int i = 0; i < 20; i++) { backend.Linear(output, input, weight, bias); }
        backend.Sync();
        for (int i = 0; i < iters; i++) { backend.Linear(output, input, weight, bias); }
        backend.Sync();

        _out.WriteLine($"Ran {iters}+20 tiny Linears at M={M},K={K},N={N}. Per-call avg ≈ per-dispatch host overhead.");

        input.Dispose(); weight.Dispose(); bias.Dispose(); output.Dispose();
    }
}
