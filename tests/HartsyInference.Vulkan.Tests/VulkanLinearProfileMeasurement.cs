using HartsyInference.Core.Backends;
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

    /// <summary>Phase-2 probe: one synthetic LLM decode-step (RmsNorm → QKV Linear → RoPE → KV-cache
    /// append → attention → out-proj → residual → RmsNorm → gate-up Linear → SwiGLU → down-proj →
    /// residual), the shape the audit flagged as the open question — does per-dispatch overhead
    /// (<see cref="Measure_TinyLinear_DispatchOverhead"/>'s finding, measured on ONE op in isolation) or
    /// GPU-residency stalls (CPU-loop-default <c>IBackend</c> members reading <c>DataPointer</c> mid-step)
    /// dominate at full-pipeline scale? Deliberately exercises both GPU-native ops (RmsNorm/Linear/
    /// Permute0213/SDPA/Add, all overridden in <see cref="VulkanBackend"/>) AND ops that fall through to
    /// <c>IBackend</c>'s CPU-loop default on Vulkan today (<c>SliceLastDim</c>, <c>ApplyRope</c>,
    /// <c>KvCacheAppend</c>, and <c>GluActivate</c>'s default composition which itself calls
    /// <c>SliceLastDim</c> twice more) — every one of those is a hidden device→host→device round trip.
    /// <see cref="VulkanBackend.GetD2hSyncCount"/> makes that hidden cost visible instead of assumed.
    /// Not a correctness gate — shapes/weights are synthetic. Run with HARTSYINFERENCE_VK_PROFILE=1 for
    /// the per-dispatch breakdown alongside the sync count and wall-clock numbers this test reports.</summary>
    [Fact]
    public unsafe void Measure_LlmDecodeStep_ResidencyVsDispatchOverhead()
    {
        const int hidden = 1024, heads = 16, headDim = 64, ff = 4096, maxSeq = 128;
        const int qkvWidth = 3 * hidden;   // fused QKV, MHA (kvHeads == heads keeps ApplyRope's shared-shape contract valid)
        const int iters = 100;
        float scale = 1f / MathF.Sqrt(headDim);

        // IBackend-typed (not VulkanBackend) so the default-interface-method ops below (SliceLastDim,
        // ApplyRope, KvCacheAppend, GluActivate — none overridden in VulkanBackend, see class doc) actually
        // resolve: C# only dispatches default interface methods through the interface type.
        using IBackend backend = new VulkanBackend();

        Tensor x = new(new TensorShape(1, 1, hidden), DType.F32);
        Tensor normW = new(new TensorShape(hidden), DType.F32);
        Tensor wQkv = new(new TensorShape(qkvWidth, hidden), DType.F32);
        Tensor qkv = new(new TensorShape(1, 1, qkvWidth), DType.F32);
        Tensor normed = new(new TensorShape(1, 1, hidden), DType.F32);
        Tensor q = new(new TensorShape(1, 1, heads, headDim), DType.F32);
        Tensor k = new(new TensorShape(1, 1, heads, headDim), DType.F32);
        Tensor v = new(new TensorShape(1, 1, heads, headDim), DType.F32);
        Tensor cos = new(new TensorShape(1, 1, headDim), DType.F32);
        Tensor sin = new(new TensorShape(1, 1, headDim), DType.F32);
        Tensor qP = new(new TensorShape(1, heads, 1, headDim), DType.F32);
        Tensor kP = new(new TensorShape(1, heads, 1, headDim), DType.F32);
        Tensor vP = new(new TensorShape(1, heads, 1, headDim), DType.F32);
        Tensor kCache = new(new TensorShape(1, heads, maxSeq, headDim), DType.F32);
        Tensor vCache = new(new TensorShape(1, heads, maxSeq, headDim), DType.F32);
        Tensor attnOut = new(new TensorShape(1, heads, 1, headDim), DType.F32);
        Tensor attnMerged = new(new TensorShape(1, 1, hidden), DType.F32);
        Tensor wOut = new(new TensorShape(hidden, hidden), DType.F32);
        Tensor projOut = new(new TensorShape(1, 1, hidden), DType.F32);
        Tensor resid1 = new(new TensorShape(1, 1, hidden), DType.F32);
        Tensor normW2 = new(new TensorShape(hidden), DType.F32);
        Tensor normed2 = new(new TensorShape(1, 1, hidden), DType.F32);
        Tensor wGateUp = new(new TensorShape(2 * ff, hidden), DType.F32);
        Tensor gateUp = new(new TensorShape(1, 1, 2 * ff), DType.F32);
        Tensor glu = new(new TensorShape(1, 1, ff), DType.F32);
        Tensor wDown = new(new TensorShape(hidden, ff), DType.F32);
        Tensor downOut = new(new TensorShape(1, 1, hidden), DType.F32);
        Tensor resid2 = new(new TensorShape(1, 1, hidden), DType.F32);

        static void FillDeterministic(Tensor t, int seed)
        {
            float* p = (float*)t.DataPointer;
            long n = t.ElementCount;
            for (long i = 0; i < n; i++) p[i] = ((((i + seed) * 2654435761L) % 1000) / 1000f - 0.5f) * 0.1f;
        }
        FillDeterministic(x, 1); FillDeterministic(normW, 2); FillDeterministic(wQkv, 3);
        FillDeterministic(cos, 4); FillDeterministic(sin, 5); FillDeterministic(wOut, 6);
        FillDeterministic(normW2, 7); FillDeterministic(wGateUp, 8); FillDeterministic(wDown, 9);
        // cos/sin should look like a real rotary table (bounded [-1,1]), not the same tiny-noise fill.
        float* cp = (float*)cos.DataPointer, sp = (float*)sin.DataPointer;
        for (int i = 0; i < headDim; i++) { cp[i] = MathF.Cos(i * 0.1f); sp[i] = MathF.Sin(i * 0.1f); }

        backend.PreloadWeights(new[] { normW, wQkv, wOut, normW2, wGateUp, wDown });

        void RunStep()
        {
            backend.RmsNorm(normed, x, normW, 1e-6f);
            backend.Linear(qkv, normed, wQkv, null);
            backend.SliceLastDim(q, qkv, 0);
            backend.SliceLastDim(k, qkv, hidden);
            backend.SliceLastDim(v, qkv, 2 * hidden);
            backend.ApplyRope(q, k, cos, sin);
            backend.Permute0213(qP, q, 1, heads, headDim);
            backend.Permute0213(kP, k, 1, heads, headDim);
            backend.Permute0213(vP, v, 1, heads, headDim);
            backend.KvCacheAppend(kCache, kP, 0);
            backend.KvCacheAppend(vCache, vP, 0);
            backend.ScaledDotProductAttention(attnOut, qP, kP, vP, null, scale);
            backend.CopyTo(attnMerged, attnOut);   // [1,H,1,D] and [1,1,hidden] share the same flat layout when seq=1
            backend.Linear(projOut, attnMerged, wOut, null);
            backend.Add(resid1, x, projOut);
            backend.RmsNorm(normed2, resid1, normW2, 1e-6f);
            backend.Linear(gateUp, normed2, wGateUp, null);
            backend.GluActivate(glu, gateUp, ff, gelu: false);
            backend.Linear(downOut, glu, wDown, null);
            backend.Add(resid2, resid1, downOut);
        }

        // Warm up (pipeline builds), then reset the counter and measure.
        VulkanBackend vk = (VulkanBackend)backend;
        for (int i = 0; i < 5; i++) RunStep();
        backend.Sync();
        backend.ResetD2hSyncCount();
        (long hitsBefore, long missesBefore) = vk.GetTransferCacheStats();
        System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) RunStep();
        backend.Sync();
        sw.Stop();
        long syncs = backend.GetD2hSyncCount();
        (long hitsAfter, long missesAfter) = vk.GetTransferCacheStats();
        long misses = missesAfter - missesBefore;

        _out.WriteLine(
            $"Ran {iters} synthetic LLM decode steps (hidden={hidden}, heads={heads}, headDim={headDim}, ff={ff}). " +
            $"Total {sw.Elapsed.TotalMilliseconds:F2} ms ({sw.Elapsed.TotalMilliseconds / iters:F3} ms/step). " +
            $"D2H syncs: {syncs} total ({(double)syncs / iters:F1}/step) — every one is a CPU-loop-default IBackend " +
            "member (SliceLastDim/ApplyRope/KvCacheAppend/CopyTo/GluActivate's internal SliceLastDim) forcing a " +
            $"device stall + readback. Transfer-cache misses (fresh H2D uploads): {misses} total ({(double)misses / iters:F1}/step) " +
            $"— hits: {hitsAfter - hitsBefore} total ({(double)(hitsAfter - hitsBefore) / iters:F1}/step). Misses are the OTHER " +
            "half of the same residency break (a tensor synced to host for a CPU-default op must be re-uploaded the next " +
            "time a GPU op needs it) and are invisible to the D2H sync count alone. Profile dump (if " +
            "HARTSYINFERENCE_VK_PROFILE=1) on backend dispose shows the per-op dispatch breakdown to compare against both.");

        x.Dispose(); normW.Dispose(); wQkv.Dispose(); qkv.Dispose(); normed.Dispose();
        q.Dispose(); k.Dispose(); v.Dispose(); cos.Dispose(); sin.Dispose();
        qP.Dispose(); kP.Dispose(); vP.Dispose(); kCache.Dispose(); vCache.Dispose();
        attnOut.Dispose(); attnMerged.Dispose(); wOut.Dispose(); projOut.Dispose(); resid1.Dispose();
        normW2.Dispose(); normed2.Dispose(); wGateUp.Dispose(); gateUp.Dispose(); glu.Dispose();
        wDown.Dispose(); downOut.Dispose(); resid2.Dispose();
    }
}
