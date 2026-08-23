using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>A SDPA additive bias that depends only on the KEY is stored as one <c>[1, Skv]</c> row and broadcast,
/// instead of the <c>[Sq, Skv]</c> duplicate of it. Wan-Animate-2's distillation build is the caller: its
/// <c>log_scale</c> band has no query-side condition, and the duplicate form forced every masked call onto the
/// materialized <c>[heads, Sq, Skv]</c> score matrix — 5836 MB at 480x800 / 61 frames.
///
/// Every test here is an equality against the duplicate form, which is the previous behaviour. If the broadcast
/// is wired to the wrong axis, dropped, or applied to the wrong key range, the bias stops matching and these
/// fail — the vector form producing SOMETHING is not the property under test.</summary>
[Collection("CudaSerial")]
public sealed unsafe class KeyOnlySdpaBiasTests
{
    private readonly ITestOutputHelper _output;
    public KeyOnlySdpaBiasTests(ITestOutputHelper output) => _output = output;

    private static uint _rng = 0x2F6E19A3u;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return (_rng & 0xFFFF) / 65535f - 0.5f; }

    private static Tensor Rnd(int a, int b, int c, int d)
    {
        Tensor t = new(new TensorShape(a, b, c, d), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = Rand();
        return t;
    }

    /// <summary>The <c>[Sq, Skv]</c> duplicate of a key-only bias row — what the engine used to be handed.</summary>
    private static Tensor Expand(Tensor keyRow, int sq)
    {
        int skv = (int)keyRow.Shape[1];
        Tensor full = new(new TensorShape(sq, skv), DType.F32);
        float* src = (float*)keyRow.DataPointer, dst = (float*)full.DataPointer;
        for (int q = 0; q < sq; q++)
            for (int k = 0; k < skv; k++) dst[(long)q * skv + k] = src[k];
        return full;
    }

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(dir)
            ? dir
            : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }

    private bool Skip()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return true; }
        CudnnRuntime.EnsureProbed();
        return false;
    }

    private static float MaxDiff(Tensor a, Tensor b)
    {
        float* pa = (float*)a.DataPointer, pb = (float*)b.DataPointer;
        float worst = 0f;
        for (long i = 0; i < a.ElementCount; i++) worst = MathF.Max(worst, MathF.Abs(pa[i] - pb[i]));
        return worst;
    }

    /// <summary>Runs one masked SDPA under a fixed set of dispatch env switches and returns the output.</summary>
    private static Tensor Run(Tensor q, Tensor k, Tensor v, Tensor mask, float scale, (string Key, string? Value)[] env)
    {
        (string Key, string? Value)[] saved = env.Select(e => (e.Key, Environment.GetEnvironmentVariable(e.Key))).ToArray();
        foreach ((string key, string? value) in env) Environment.SetEnvironmentVariable(key, value);
        try
        {
            using CudaBackend backend = new(0, PtxDir());
            Tensor outT = new(q.Shape, DType.F32);
            ((IBackend)backend).ScaledDotProductAttention(outT, q, k, v, mask, scale, allowF16: true);
            backend.Sync();
            _ = outT.DataPointer;   // force the readback before the backend goes away
            return outT;
        }
        finally
        {
            foreach ((string key, string? value) in saved) Environment.SetEnvironmentVariable(key, value);
        }
    }

    /// <summary>Wan-Animate-2's real bias — the <c>[hw, 2hw)</c> key band at <c>log_scale = -1.3</c> — broadcast
    /// from one row must match the duplicate form on the fused cuDNN path.</summary>
    [Fact]
    public void CudnnFusedPath_KeyOnlyBias_MatchesTheDuplicate()
    {
        if (Skip()) return;
        if (!CudnnRuntime.SupportsSdpa) { _output.WriteLine($"SKIPPED: {CudnnRuntime.Reason}"); return; }

        const int heads = 4, hw = 128, frames = 4, d = 128;
        const int sq = hw, skv = hw * frames;
        float scale = 1f / MathF.Sqrt(d);
        using Tensor q = Rnd(1, heads, sq, d);
        using Tensor k = Rnd(1, heads, skv, d);
        using Tensor v = Rnd(1, heads, skv, d);
        using Tensor row = WanAnimate2Transformer.BuildLogScaleBias(hw, skv, -1.3f);
        Assert.Equal(new TensorShape(1, skv), row.Shape);
        using Tensor full = Expand(row, sq);

        (string, string?)[] cudnn = [("HARTSY_SDPA_CUDNN", "1"), ("HARTSY_SDPA_FORCE_TILED", null)];
        using Tensor broadcast = Run(q, k, v, row, scale, cudnn);
        using Tensor duplicate = Run(q, k, v, full, scale, cudnn);

        float diff = MaxDiff(broadcast, duplicate);
        _output.WriteLine($"cuDNN fused: max|broadcast - duplicate| = {diff:E3}");
        Assert.True(diff < 1e-5f, $"broadcast bias diverged from the duplicate by {diff:E3}.");
    }

    /// <summary>The memory-bounded query-tiled path is what catches a masked call whose score matrix does not fit.
    /// It must apply the same bias — the rank-1 accumulate is a different mechanism from cuDNN's score modifier.</summary>
    [Fact]
    public void TiledPath_KeyOnlyBias_MatchesTheCudnnResult()
    {
        if (Skip()) return;
        if (!CudnnRuntime.SupportsSdpa) { _output.WriteLine($"SKIPPED: {CudnnRuntime.Reason}"); return; }

        const int heads = 4, hw = 96, frames = 5, d = 128;
        const int sq = hw, skv = hw * frames;
        float scale = 1f / MathF.Sqrt(d);
        using Tensor q = Rnd(1, heads, sq, d);
        using Tensor k = Rnd(1, heads, skv, d);
        using Tensor v = Rnd(1, heads, skv, d);
        using Tensor row = WanAnimate2Transformer.BuildLogScaleBias(hw, skv, -1.3f);

        using Tensor tiled = Run(q, k, v, row, scale,
            [("HARTSY_SDPA_CUDNN", "0"), ("HARTSY_SDPA_FORCE_TILED", "1")]);
        using Tensor fused = Run(q, k, v, row, scale,
            [("HARTSY_SDPA_CUDNN", "1"), ("HARTSY_SDPA_FORCE_TILED", null)]);

        // The tiled path is TF32/F32 throughout; the fused path is fp16 I/O. Tolerance is the fp16 gap, not the
        // bias's — an unapplied or misplaced bias moves softmax weights by e^1.3 and lands orders of magnitude out.
        float diff = MaxDiff(tiled, fused);
        _output.WriteLine($"tiled vs cuDNN: max|Δ| = {diff:E3}");
        Assert.True(diff < 2e-3f, $"tiled key-only bias diverged from the fused path by {diff:E3}.");
    }

    /// <summary>The materialized GEMM path indexes the mask per query row, so it rebuilds the duplicate. Small
    /// shapes still reach it (the tiled branch only takes over when the score matrix stops fitting).</summary>
    [Fact]
    public void MaterializedPath_KeyOnlyBias_MatchesTheDuplicate()
    {
        if (Skip()) return;

        const int heads = 4, hw = 32, frames = 4, d = 64;
        const int sq = hw, skv = hw * frames;
        float scale = 1f / MathF.Sqrt(d);
        using Tensor q = Rnd(1, heads, sq, d);
        using Tensor k = Rnd(1, heads, skv, d);
        using Tensor v = Rnd(1, heads, skv, d);
        using Tensor row = WanAnimate2Transformer.BuildLogScaleBias(hw, skv, -1.3f);
        using Tensor full = Expand(row, sq);

        (string, string?)[] materialized =
            [("HARTSY_SDPA_CUDNN", "0"), ("HARTSY_SDPA_FORCE_TILED", null), ("HARTSY_SDPA_NO_F16", "1")];
        using Tensor broadcast = Run(q, k, v, row, scale, materialized);
        using Tensor duplicate = Run(q, k, v, full, scale, materialized);

        float diff = MaxDiff(broadcast, duplicate);
        _output.WriteLine($"materialized: max|broadcast - duplicate| = {diff:E3}");
        Assert.True(diff < 1e-5f, $"materialized key-only bias diverged from the duplicate by {diff:E3}.");
    }

    /// <summary>The tiled path's per-head/per-batch fix: a bias that carries one row PER HEAD (not one row shared by
    /// everything) must have each head look up its own row, not always block 0's. Regression target for the bug
    /// where <c>AccumulateKeyBias</c> was called once across all heads with a single un-offset bias pointer — every
    /// head silently got head 0's row. Ground truth is the full non-broadcast duplicate ([1,heads,Sq,Skv], each
    /// head's own row repeated over every query) run through the ordinary materialized path, which indexes the
    /// mask per query row and was never at risk of this bug.</summary>
    [Fact]
    public void TiledPath_PerHeadBias_SelectsTheOwningHeadsRow()
    {
        if (Skip()) return;

        const int heads = 4, hw = 32, frames = 4, d = 64;
        const int sq = hw, skv = hw * frames;
        float scale = 1f / MathF.Sqrt(d);
        using Tensor q = Rnd(1, heads, sq, d);
        using Tensor k = Rnd(1, heads, skv, d);
        using Tensor v = Rnd(1, heads, skv, d);

        // One [Skv] row per head, each at a different log_scale — a wrong-head lookup is not accidentally masked
        // by every head sharing the same bias.
        using Tensor rowsPerHead = new(new TensorShape(1, heads, 1, skv), DType.F32);
        using Tensor duplicatePerHead = new(new TensorShape(1, heads, sq, skv), DType.F32);
        float* pRows = (float*)rowsPerHead.DataPointer;
        float* pDup = (float*)duplicatePerHead.DataPointer;
        for (int h = 0; h < heads; h++)
        {
            using Tensor row = WanAnimate2Transformer.BuildLogScaleBias(hw, skv, -0.5f * (h + 1));
            float* pRow = (float*)row.DataPointer;
            Buffer.MemoryCopy(pRow, pRows + (long)h * skv, skv * 4, skv * 4);
            for (int qi = 0; qi < sq; qi++)
                Buffer.MemoryCopy(pRow, pDup + ((long)h * sq + qi) * skv, skv * 4, skv * 4);
        }

        (string, string?)[] tiled = [("HARTSY_SDPA_CUDNN", "0"), ("HARTSY_SDPA_FORCE_TILED", "1")];
        (string, string?)[] materialized =
            [("HARTSY_SDPA_CUDNN", "0"), ("HARTSY_SDPA_FORCE_TILED", null), ("HARTSY_SDPA_NO_F16", "1")];
        using Tensor viaTiled = Run(q, k, v, rowsPerHead, scale, tiled);
        using Tensor viaMaterialized = Run(q, k, v, duplicatePerHead, scale, materialized);

        float diff = MaxDiff(viaTiled, viaMaterialized);
        _output.WriteLine($"tiled per-head vs materialized duplicate: max|Δ| = {diff:E3}");
        Assert.True(diff < 1e-4f, $"tiled per-head bias diverged from the materialized ground truth by {diff:E3}.");
    }

    /// <summary>A VRAM shortfall inside the fused path must never disable it permanently. The fallback it demotes
    /// to allocates the whole <c>[heads, Sq, Skv]</c> score matrix — strictly MORE memory than the allocation that
    /// just failed — so treating an OOM as structural guarantees the next call fails harder.</summary>
    [Fact]
    public void OutOfVram_IsTransient_NotAStructuralKill()
    {
        if (Skip()) return;
        if (!CudnnRuntime.SupportsSdpa) { _output.WriteLine($"SKIPPED: {CudnnRuntime.Reason}"); return; }

        const int heads = 4, s = 64, d = 64;
        float scale = 1f / MathF.Sqrt(d);
        string? prev = Environment.GetEnvironmentVariable("HARTSY_SDPA_CUDNN");
        Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", "1");
        try
        {
            using CudaBackend backend = new(0, PtxDir());
            using Tensor q = Rnd(1, heads, s, d);
            using Tensor k = Rnd(1, heads, s, d);
            using Tensor v = Rnd(1, heads, s, d);
            using Tensor outT = new(new TensorShape(1, heads, s, d), DType.F32);

            int injections = 0;
            backend.TestCudnnSdpaFaultInjector = _ => { injections++; return new OutOfVramException(322L << 20, 79L << 20); };
            ((IBackend)backend).ScaledDotProductAttention(outT, q, k, v, null, scale, allowF16: true);

            Assert.Equal(1, injections);
            Assert.True(backend.CudnnSdpaDimDiagnostics.TryGetValue(
                d, out (int ConsecutiveFailures, bool Permanent, DateTimeOffset? NextRetryAt) diag));
            Assert.False(diag.Permanent, "an out-of-VRAM failure was classified as a structural cuDNN kill.");
            Assert.Equal(1, diag.ConsecutiveFailures);

            // Backoff, not a kill: once the first window (2000ms) elapses the dim is eligible again and a fresh
            // attempt is actually made. Same real-clock convention as CudnnSdpaRetryTests.
            System.Threading.Thread.Sleep(2200);
            backend.TestCudnnSdpaFaultInjector = null;
            ((IBackend)backend).ScaledDotProductAttention(outT, q, k, v, null, scale, allowF16: true);
            Assert.True(backend.CudnnSdpaEngaged, "the fused path did not recover after a transient VRAM failure.");
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", prev);
        }
    }
}
