using System.Diagnostics;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Diagnostic throughput comparison for <c>matmul_coopmat_blocked.comp.glsl</c> (register-blocked,
/// 2026-07-31) against the existing naive <c>matmul_coopmat.comp.glsl</c> (one accumulator per subgroup,
/// direct global-memory load) — the actual question this kernel exists to answer: does register blocking
/// (ggml/llama.cpp's core GEMM optimization technique — see docs/Checklists/TROUBLESHOOTING.md) close a
/// meaningful fraction of the ~30-160x Vulkan-vs-CUDA gap measured in benchmarks/scoreboards/VULKAN.md's
/// GEMM table? Uses the SAME shapes as that table for direct comparability. Not a correctness test (see
/// VulkanBackendSmokeTests.Backend_CoopmatBlocked_Diagnostic_MatchesCpu for that) — this only measures
/// wall-clock, with a real GPU sync (not just dispatch-submission time) around each timed batch.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanCoopmatBlockedBenchmark
{
    private readonly ITestOutputHelper _output;
    public VulkanCoopmatBlockedBenchmark(ITestOutputHelper output) => _output = output;

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    [Fact]
    public void Compare_Naive_Vs_Blocked_Coopmat_Throughput()
    {
        if (!VulkanAvailable()) { _output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16 || !backend.Vk.HasCooperativeMatrix)
        {
            _output.WriteLine("SKIPPED: no F16/coopmat support");
            return;
        }

        // Same shapes as benchmarks/scoreboards/VULKAN.md's GEMM table, all exact multiples of 16 (and of
        // 64, so the blocked kernel's fixed BM=BN=64 tile is fully utilized with no boundary handling in
        // play — the cleanest possible apples-to-apples comparison).
        (int M, int K, int N, string label)[] shapes =
        {
            (4096, 1280, 1280, "SDXL UNet QKV"),
            (1024, 3072, 9216, "Flux DiT QKV"),
            (1024, 3072, 12288, "Flux DiT FFN"),
            (1024, 1536, 4608, "SD3.5 joint-attn"),
            (1024, 3072, 9216, "Hunyuan Image 2.1"),
        };

        const int WarmupIters = 3;
        const int TimedIters = 10;

        // Three-way comparison isolates WHY register blocking didn't help on first measurement:
        // - naive: matmul_coopmat.comp.glsl (1 accumulator/subgroup, direct global-memory coopMatLoad, no
        //   shared memory, no barriers, 16 subgroups/workgroup at BM=BN=64).
        // - staged-only (WM=WN=16): matmul_coopmat_blocked.comp.glsl with register blocking DISABLED
        //   (cmsPerRow=cmsPerCol=1, matching naive's 1-accumulator/subgroup AND its 16-subgroups/workgroup
        //   occupancy) but STILL using shared-memory staging + barriers. Isolates "does staging alone
        //   help/hurt" from "does reducing subgroup count (register blocking) help/hurt".
        // - blocked (WM=WN=32): the original register-blocked config (4 accumulators/subgroup, only 4
        //   subgroups/workgroup).
        _output.WriteLine($"{"Shape",-40} {"Naive(us)",10} {"Staged(us)",11} {"Blocked(us)",12} {"Staged/Naive",13} {"Blocked/Naive",14}");
        foreach ((int M, int K, int N, string label) in shapes)
        {
            Tensor input = new(new TensorShape(M, K), DType.F16);
            Tensor weight = new(new TensorShape(N, K), DType.F16);
            Tensor outNaive = new(new TensorShape(M, N), DType.F16);
            Tensor outStaged = new(new TensorShape(M, N), DType.F16);
            Tensor outBlocked = new(new TensorShape(M, N), DType.F16);

            Random rng = new(42);
            Span<Half> iS = input.AsSpan<Half>();
            Span<Half> wS = weight.AsSpan<Half>();
            for (int i = 0; i < M * K; i++) iS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);
            for (int i = 0; i < N * K; i++) wS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);

            for (int i = 0; i < WarmupIters; i++)
            {
                backend.Linear(outNaive, input, weight, null);
                backend.TryDispatchCoopmatBlockedDiagnostic(outStaged, input, weight, transposeA: false, transposeB: true, null, wm: 16, wn: 16);
                backend.TryDispatchCoopmatBlockedDiagnostic(outBlocked, input, weight, transposeA: false, transposeB: true, null, wm: 32, wn: 32);
            }
            backend.Sync();

            Stopwatch swNaive = Stopwatch.StartNew();
            for (int i = 0; i < TimedIters; i++) backend.Linear(outNaive, input, weight, null);
            backend.Sync();
            swNaive.Stop();

            Stopwatch swStaged = Stopwatch.StartNew();
            for (int i = 0; i < TimedIters; i++) backend.TryDispatchCoopmatBlockedDiagnostic(outStaged, input, weight, transposeA: false, transposeB: true, null, wm: 16, wn: 16);
            backend.Sync();
            swStaged.Stop();

            Stopwatch swBlocked = Stopwatch.StartNew();
            for (int i = 0; i < TimedIters; i++) backend.TryDispatchCoopmatBlockedDiagnostic(outBlocked, input, weight, transposeA: false, transposeB: true, null, wm: 32, wn: 32);
            backend.Sync();
            swBlocked.Stop();

            double naiveUs = swNaive.Elapsed.TotalMicroseconds / TimedIters;
            double stagedUs = swStaged.Elapsed.TotalMicroseconds / TimedIters;
            double blockedUs = swBlocked.Elapsed.TotalMicroseconds / TimedIters;
            _output.WriteLine($"({M},{K},{N}) {label,-22} {naiveUs,10:F1} {stagedUs,11:F1} {blockedUs,12:F1} {naiveUs / stagedUs,12:F2}x {naiveUs / blockedUs,13:F2}x");

            input.Dispose(); weight.Dispose(); outNaive.Dispose(); outStaged.Dispose(); outBlocked.Dispose();
        }
    }

    /// <summary>Same 3-way comparison as <see cref="Compare_Naive_Vs_Blocked_Coopmat_Throughput"/>, but using
    /// <c>VulkanBackend.MeasureGpuTimeMs</c> (a real <c>VkQueryPool</c> GPU timestamp region, not a host
    /// <c>Stopwatch</c>) — isolates PURE GPU execution time from host submission/dispatch/wait overhead, and
    /// is directly comparable in absolute terms to benchmarks/scoreboards/VULKAN.md's CUDA column (both are
    /// GPU-execution-time measurements, unlike the batched-Stopwatch numbers above, which mix in host
    /// overhead the CUDA number's methodology may not include the same way). Built after Nsight Compute was
    /// confirmed CUDA-only (no Vulkan support in its own --help output) and Nsight Graphics wasn't
    /// apt-installable on this box — see docs/Checklists/TROUBLESHOOTING.md.</summary>
    [Fact]
    public void Compare_Naive_Vs_Blocked_Coopmat_GpuOnlyTime()
    {
        if (!VulkanAvailable()) { _output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16 || !backend.Vk.HasCooperativeMatrix)
        {
            _output.WriteLine("SKIPPED: no F16/coopmat support");
            return;
        }

        // (M, K, N, label, documented CUDA time in us — from benchmarks/scoreboards/VULKAN.md's GEMM table)
        (int M, int K, int N, string label, double cudaUs)[] shapes =
        {
            (4096, 1280, 1280, "SDXL UNet QKV", 187.3),
            (1024, 3072, 9216, "Flux DiT QKV", 463.7),
            (1024, 3072, 12288, "Flux DiT FFN", 655.1),
            (1024, 1536, 4608, "SD3.5 joint-attn", 219.2),
            (1024, 3072, 9216, "Hunyuan Image 2.1", 528.9),
        };

        const int WarmupIters = 5;
        const int TimedIters = 20;

        _output.WriteLine($"{"Shape",-40} {"CUDA(us)",9} {"Naive(us)",10} {"Staged(us)",11} {"Blocked(us)",12} {"Naive/CUDA",11} {"Staged/CUDA",12}");
        foreach ((int M, int K, int N, string label, double cudaUs) in shapes)
        {
            Tensor input = new(new TensorShape(M, K), DType.F16);
            Tensor weight = new(new TensorShape(N, K), DType.F16);
            Tensor outNaive = new(new TensorShape(M, N), DType.F16);
            Tensor outStaged = new(new TensorShape(M, N), DType.F16);
            Tensor outBlocked = new(new TensorShape(M, N), DType.F16);

            Random rng = new(42);
            Span<Half> iS = input.AsSpan<Half>();
            Span<Half> wS = weight.AsSpan<Half>();
            for (int i = 0; i < M * K; i++) iS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);
            for (int i = 0; i < N * K; i++) wS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);

            for (int i = 0; i < WarmupIters; i++)
            {
                backend.Linear(outNaive, input, weight, null);
                backend.TryDispatchCoopmatBlockedDiagnostic(outStaged, input, weight, transposeA: false, transposeB: true, null, wm: 16, wn: 16);
                backend.TryDispatchCoopmatBlockedDiagnostic(outBlocked, input, weight, transposeA: false, transposeB: true, null, wm: 32, wn: 32);
            }
            backend.Sync();

            double naiveMs = backend.MeasureGpuTimeMs(TimedIters, () => backend.Linear(outNaive, input, weight, null));
            double stagedMs = backend.MeasureGpuTimeMs(TimedIters, () => backend.TryDispatchCoopmatBlockedDiagnostic(outStaged, input, weight, transposeA: false, transposeB: true, null, wm: 16, wn: 16));
            double blockedMs = backend.MeasureGpuTimeMs(TimedIters, () => backend.TryDispatchCoopmatBlockedDiagnostic(outBlocked, input, weight, transposeA: false, transposeB: true, null, wm: 32, wn: 32));

            double naiveUs = naiveMs * 1000.0 / TimedIters;
            double stagedUs = stagedMs * 1000.0 / TimedIters;
            double blockedUs = blockedMs * 1000.0 / TimedIters;
            _output.WriteLine($"({M},{K},{N}) {label,-22} {cudaUs,9:F1} {naiveUs,10:F1} {stagedUs,11:F1} {blockedUs,12:F1} {naiveUs / cudaUs,10:F1}x {stagedUs / cudaUs,11:F1}x");

            input.Dispose(); weight.Dispose(); outNaive.Dispose(); outStaged.Dispose(); outBlocked.Dispose();
        }
    }

    /// <summary>The actual question this session's low-level Vulkan work exists to answer: does
    /// <c>VK_NV_cooperative_matrix2</c> (workgroup-scope, tensor-layout-addressed, hardware-clamped GEMM —
    /// architecturally distinct from coopmat1's subgroup-scope fixed 16x16x16 fragments) close any of the
    /// real Vulkan-vs-CUDA GEMM gap that coopmat1 was measured NOT to close (coopmat1 == scalar-fallback
    /// throughput on this RTX 4090; see docs/Checklists/TROUBLESHOOTING.md's 2026-07-31 GPU-profiling
    /// entry)? Uses <c>VulkanBackend.MeasureGpuTimeMs</c> (real <c>VkQueryPool</c> GPU timestamps, not host
    /// Stopwatch) so the result is directly comparable in absolute terms to the documented CUDA column.</summary>
    [Fact]
    public void Compare_Naive_Vs_CoopMat2_GpuOnlyTime()
    {
        if (!VulkanAvailable()) { _output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16 || !backend.Vk.HasCooperativeMatrix)
        {
            _output.WriteLine("SKIPPED: no F16/coopmat support");
            return;
        }
        if (!backend.Vk.HasCooperativeMatrix2)
        {
            _output.WriteLine("SKIPPED: no VK_NV_cooperative_matrix2 support (NVIDIA-only extension)");
            return;
        }

        // (M, K, N, label, documented CUDA time in us — from benchmarks/scoreboards/VULKAN.md's GEMM table)
        (int M, int K, int N, string label, double cudaUs)[] shapes =
        {
            (4096, 1280, 1280, "SDXL UNet QKV", 187.3),
            (1024, 3072, 9216, "Flux DiT QKV", 463.7),
            (1024, 3072, 12288, "Flux DiT FFN", 655.1),
            (1024, 1536, 4608, "SD3.5 joint-attn", 219.2),
            (1024, 3072, 9216, "Hunyuan Image 2.1", 528.9),
        };

        const int WarmupIters = 5;
        const int TimedIters = 20;

        _output.WriteLine($"{"Shape",-40} {"CUDA(us)",9} {"Naive(us)",10} {"CoopMat2(us)",13} {"Naive/CUDA",11} {"CoopMat2/CUDA",14} {"CoopMat2/Naive",15}");
        foreach ((int M, int K, int N, string label, double cudaUs) in shapes)
        {
            Tensor input = new(new TensorShape(M, K), DType.F16);
            Tensor weight = new(new TensorShape(N, K), DType.F16);
            Tensor outNaive = new(new TensorShape(M, N), DType.F16);
            Tensor outCoopMat2 = new(new TensorShape(M, N), DType.F16);

            Random rng = new(42);
            Span<Half> iS = input.AsSpan<Half>();
            Span<Half> wS = weight.AsSpan<Half>();
            for (int i = 0; i < M * K; i++) iS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);
            for (int i = 0; i < N * K; i++) wS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);

            for (int i = 0; i < WarmupIters; i++)
            {
                backend.Linear(outNaive, input, weight, null);
                backend.TryDispatchCoopMat2(outCoopMat2, input, weight, transposeA: false, transposeB: true, bias: null);
            }
            backend.Sync();

            double naiveMs = backend.MeasureGpuTimeMs(TimedIters, () => backend.Linear(outNaive, input, weight, null));
            double coopMat2Ms = backend.MeasureGpuTimeMs(TimedIters, () => backend.TryDispatchCoopMat2(outCoopMat2, input, weight, transposeA: false, transposeB: true, bias: null));

            double naiveUs = naiveMs * 1000.0 / TimedIters;
            double coopMat2Us = coopMat2Ms * 1000.0 / TimedIters;
            _output.WriteLine($"({M},{K},{N}) {label,-22} {cudaUs,9:F1} {naiveUs,10:F1} {coopMat2Us,13:F1} {naiveUs / cudaUs,10:F1}x {coopMat2Us / cudaUs,13:F1}x {coopMat2Us / naiveUs,14:F2}x");

            input.Dispose(); weight.Dispose(); outNaive.Dispose(); outCoopMat2.Dispose();
        }
    }

    /// <summary>Re-runs the coopmat2-vs-coopmat1 comparison against the CORRECT competitor after the real
    /// Krea2 e2e run (2026-07-31) found the earlier <see cref="Compare_Naive_Vs_CoopMat2_GpuOnlyTime"/>
    /// result didn't hold up: that benchmark only used M values that are exact multiples of 16, so coopmat1
    /// always took <c>matmul_coopmat</c> (the plain aligned-only kernel) — but on the real model, 60.7% of
    /// coopmat1's engagement goes through <c>matmul_coopmat_partial_m</c> (the shared-memory-staged
    /// non-aligned-M variant, itself already a real win over the plain scalar path), which coopmat2 was
    /// never actually benchmarked against. This test compares coopmat2 against BOTH: the same M values as
    /// before (forces <c>matmul_coopmat</c>) and those values +3 (forces <c>matmul_coopmat_partial_m</c> —
    /// N, K stay 16-aligned so only M's alignment changes which coopmat1 kernel engages). Goes through the
    /// REAL <see cref="VulkanBackend.Linear"/>/<c>DispatchMatmul</c> path (not the raw kernel entry points)
    /// so bias-epilogue cost is included exactly as it is in production — measured both with and without
    /// bias since both kernels fuse it (coopmat2's fusion landed 2026-07-31, same day as this test).</summary>
    [Fact]
    public void Compare_CoopMat1PartialM_Vs_CoopMat2_GpuOnlyTime()
    {
        if (!VulkanAvailable()) { _output.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16 || !backend.Vk.HasCooperativeMatrix)
        {
            _output.WriteLine("SKIPPED: no F16/coopmat support");
            return;
        }
        if (!backend.Vk.HasCooperativeMatrix2)
        {
            _output.WriteLine("SKIPPED: no VK_NV_cooperative_matrix2 support (NVIDIA-only extension)");
            return;
        }

        (int M, int K, int N, string label)[] shapes =
        {
            (4096, 1280, 1280, "SDXL UNet QKV"),
            (1024, 3072, 9216, "Flux DiT QKV"),
            (1024, 3072, 12288, "Flux DiT FFN"),
            (1024, 1536, 4608, "SD3.5 joint-attn"),
            (1024, 3072, 9216, "Hunyuan Image 2.1"),
        };

        const int WarmupIters = 5;
        const int TimedIters = 20;

        _output.WriteLine($"{"Shape",-38} {"MAligned",8} {"Bias",5} {"CoopMat1(us)",13} {"CoopMat2(us)",13} {"CoopMat2/CoopMat1",18}");
        foreach ((int M, int K, int N, string label) in shapes)
        {
            foreach (bool mAligned in new[] { true, false })
            foreach (bool hasBias in new[] { false, true })
            {
                int mUse = mAligned ? M : M + 3;   // +3 forces matmul_coopmat_partial_m for coopmat1
                Tensor input = new(new TensorShape(mUse, K), DType.F16);
                Tensor weight = new(new TensorShape(N, K), DType.F16);
                Tensor? bias = hasBias ? new Tensor(new TensorShape(N), DType.F16) : null;
                Tensor outCoopMat1 = new(new TensorShape(mUse, N), DType.F16);
                Tensor outCoopMat2 = new(new TensorShape(mUse, N), DType.F16);

                Random rng = new(42);
                Span<Half> iS = input.AsSpan<Half>();
                Span<Half> wS = weight.AsSpan<Half>();
                for (int i = 0; i < mUse * K; i++) iS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);
                for (int i = 0; i < N * K; i++) wS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);
                if (hasBias)
                {
                    Span<Half> bS = bias!.AsSpan<Half>();
                    for (int i = 0; i < N; i++) bS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.05f);
                }

                backend.EnableCoopMat2 = false;
                for (int i = 0; i < WarmupIters; i++) backend.Linear(outCoopMat1, input, weight, bias);
                backend.Sync();
                double coopMat1Ms = backend.MeasureGpuTimeMs(TimedIters, () => backend.Linear(outCoopMat1, input, weight, bias));

                backend.EnableCoopMat2 = true;
                for (int i = 0; i < WarmupIters; i++) backend.Linear(outCoopMat2, input, weight, bias);
                backend.Sync();
                double coopMat2Ms = backend.MeasureGpuTimeMs(TimedIters, () => backend.Linear(outCoopMat2, input, weight, bias));
                backend.EnableCoopMat2 = false;

                double coopMat1Us = coopMat1Ms * 1000.0 / TimedIters;
                double coopMat2Us = coopMat2Ms * 1000.0 / TimedIters;
                _output.WriteLine($"({mUse},{K},{N}) {label,-22} {(mAligned ? "yes" : "no"),8} {(hasBias ? "yes" : "no"),5} {coopMat1Us,13:F1} {coopMat2Us,13:F1} {coopMat2Us / coopMat1Us,17:F2}x");

                input.Dispose(); weight.Dispose(); bias?.Dispose(); outCoopMat1.Dispose(); outCoopMat2.Dispose();
            }
        }
    }

    /// <summary>Isolates the ~4-second one-time stall observed in real Krea2 e2e runs with coopmat2 opted
    /// in — every benchmark above warms up (5 iterations) BEFORE timing, which would silently absorb a
    /// genuine first-use cost. This measures wall-clock (Stopwatch, matching how the real run's host-side
    /// profiler would see it — NOT <c>MeasureGpuTimeMs</c>, which explicitly excludes host/dispatch/wait
    /// overhead and would hide exactly this kind of cost) for dispatch #1 vs. dispatches #2-10 of a fresh
    /// backend's FIRST-EVER coopmat2 call, and separately of a fresh backend's first-ever coopmat1 call for
    /// comparison (does the same first-use cost exist there too, or is it coopmat2-specific?).</summary>
    [Fact]
    public void Measure_CoopMat2_FirstDispatch_ColdStartLatency()
    {
        if (!VulkanAvailable()) { _output.WriteLine("SKIPPED: no Vulkan device"); return; }

        const int M = 1024, K = 3072, N = 9216;   // Flux DiT QKV shape — one of the real GEMM-scoreboard shapes

        // Fresh backend, coopmat2 path, no warmup at all — first call ever.
        using (VulkanBackend backend = new())
        {
            if (!backend.Vk.HasCooperativeMatrix2) { _output.WriteLine("SKIPPED: no coopmat2"); return; }
            backend.EnableCoopMat2 = true;

            Tensor input = new(new TensorShape(M, K), DType.F16);
            Tensor weight = new(new TensorShape(N, K), DType.F16);
            Tensor output = new(new TensorShape(M, N), DType.F16);
            Random rng = new(1);
            Span<Half> iS = input.AsSpan<Half>();
            Span<Half> wS = weight.AsSpan<Half>();
            for (int i = 0; i < M * K; i++) iS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);
            for (int i = 0; i < N * K; i++) wS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);

            _output.WriteLine("=== CoopMat2 (fresh backend, no warmup) ===");
            for (int i = 0; i < 10; i++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                backend.Linear(output, input, weight, null);
                backend.Sync();
                sw.Stop();
                _output.WriteLine($"  dispatch #{i + 1}: {sw.Elapsed.TotalMilliseconds:F1} ms");
            }
            input.Dispose(); weight.Dispose(); output.Dispose();
        }

        // Fresh backend, coopmat1 path (opt-in off), same shape — is the cold-start cost coopmat2-specific?
        using (VulkanBackend backend = new())
        {
            if (!backend.Vk.HasCooperativeMatrix) { _output.WriteLine("SKIPPED: no coopmat1"); return; }

            Tensor input = new(new TensorShape(M, K), DType.F16);
            Tensor weight = new(new TensorShape(N, K), DType.F16);
            Tensor output = new(new TensorShape(M, N), DType.F16);
            Random rng = new(1);
            Span<Half> iS = input.AsSpan<Half>();
            Span<Half> wS = weight.AsSpan<Half>();
            for (int i = 0; i < M * K; i++) iS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);
            for (int i = 0; i < N * K; i++) wS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);

            _output.WriteLine("=== CoopMat1 (fresh backend, no warmup) ===");
            for (int i = 0; i < 10; i++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                backend.Linear(output, input, weight, null);
                backend.Sync();
                sw.Stop();
                _output.WriteLine($"  dispatch #{i + 1}: {sw.Elapsed.TotalMilliseconds:F1} ms");
            }
            input.Dispose(); weight.Dispose(); output.Dispose();
        }
    }
}
