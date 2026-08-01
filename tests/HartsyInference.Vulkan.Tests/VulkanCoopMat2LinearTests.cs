using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>Gate for <c>VulkanBackend.EnableCoopMat2</c> wiring coopmat2 into the real <see cref="VulkanBackend.Linear"/>/
/// <c>DispatchMatmul</c> call path — not just the standalone kernel (see <c>VulkanBackendSmokeTests.Backend_CoopMat2_MatchesCpu</c>
/// for that). This is the thing that actually matters for real models: coopmat1 only reaches 0.8% of a
/// real Krea2 run's GEMMs (M/N/K must be EXACT multiples of 16, no host padding, and prompt-length-dependent
/// sequence lengths essentially never land on one) — coopmat2's hardware-clamped addressing has no such
/// gate, so these tests deliberately use PROMPT-REALISTIC non-16-aligned shapes (e.g. M=4109, mirroring the
/// exact jointSeq=imgSeq+txtSeq value that blocked coopmat1 on a real run — see
/// docs/Checklists/TROUBLESHOOTING.md) to prove coopmat2 actually engages where coopmat1 structurally
/// cannot, using <see cref="VulkanBackend.GemmEngagementCounts"/> as the discriminator (correctness alone
/// can't tell which path handled a dispatch, since both compute the right answer for shapes both can
/// handle).</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanCoopMat2LinearTests
{
    private readonly ITestOutputHelper _out;
    public VulkanCoopMat2LinearTests(ITestOutputHelper output) => _out = output;

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    [Fact]
    public void Backend_Linear_CoopMat2OptIn_DefaultsOff()
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        Assert.False(backend.EnableCoopMat2, "EnableCoopMat2 must default to off unless HARTSYINFERENCE_VK_COOPMAT2=1 is set.");
    }

    /// <summary>The core claim: with the opt-in enabled, a Linear call at a shape coopmat1 could NEVER
    /// reach (M not a multiple of 16 — the same alignment-defeating property as the real jointSeq=4109
    /// shape from a real Krea2 run, just scaled down so the CPU reference triple-loop below stays fast) is
    /// handled by coopmat2, not the tiled fallback. Verified via <see cref="VulkanBackend.GemmEngagementCounts"/>,
    /// not just correctness — a tiled-fallback dispatch would ALSO produce the right numeric answer, so
    /// correctness alone can't distinguish "coopmat2 engaged" from "silently fell through."</summary>
    [Theory]
    [InlineData(257, 512, 512, true)]      // M not 16-aligned — coopmat1's exact-multiple-of-16 gate blocks this
    [InlineData(13, 512, 512, false)]      // tiny M (mirrors the real txtSeq-alone shape), no bias
    [InlineData(512, 512, 512, true)]      // 16-aligned control — coopmat2 must ALSO handle the case coopmat1 could
    public void Backend_Linear_CoopMat2OptIn_EngagesAndMatchesF32Reference(int M, int K, int N, bool hasBias)
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16 || !backend.Vk.HasCooperativeMatrix2)
        {
            _out.WriteLine("SKIPPED: no F16/coopmat2 support");
            return;
        }

        Tensor input = new(new TensorShape(M, K), DType.F16);
        Tensor weight = new(new TensorShape(N, K), DType.F16);
        Tensor? bias = hasBias ? new Tensor(new TensorShape(N), DType.F16) : null;
        Tensor output = new(new TensorShape(M, N), DType.F16);
        try
        {
            Random rng = new(4000 + M + N);
            Span<Half> iS = input.AsSpan<Half>();
            Span<Half> wS = weight.AsSpan<Half>();
            Half[] bS = new Half[N];
            for (int i = 0; i < M * K; i++) iS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);
            for (int i = 0; i < N * K; i++) wS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);
            if (hasBias)
            {
                Span<Half> bSpan = bias!.AsSpan<Half>();
                for (int i = 0; i < N; i++) { bS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.05f); bSpan[i] = bS[i]; }
            }

            backend.EnableCoopMat2 = true;
            (long coopMat2Before, _, _) = backend.GemmEngagementCounts;
            backend.Linear(output, input, weight, bias);
            (long coopMat2After, long coopMatAfter, long tiledAfter) = backend.GemmEngagementCounts;

            Assert.True(coopMat2After == coopMat2Before + 1,
                $"Expected exactly one coopmat2 dispatch for M={M},K={K},N={N} — got coopmat2={coopMat2After - coopMat2Before}, " +
                $"coopmat1={coopMatAfter}, tiled={tiledAfter}. coopmat2 should engage unconditionally at this shape " +
                "(no M/N/K alignment requirement) once opted in.");

            ReadOnlySpan<Half> oS = output.AsReadOnlySpan<Half>();
            float maxRel = 0f; int firstM = -1, firstN = -1;
            for (int m = 0; m < M; m++)
                for (int n = 0; n < N; n++)
                {
                    float acc = hasBias ? (float)bS[n] : 0f;
                    for (int k = 0; k < K; k++) acc += (float)iS[m * K + k] * (float)wS[n * K + k];
                    float rel = MathF.Abs((float)oS[m * N + n] - acc) / MathF.Max(1e-3f, MathF.Abs(acc));
                    if (rel > maxRel) { maxRel = rel; firstM = m; firstN = n; }
                }
            _out.WriteLine($"Linear (M={M},K={K},N={N},bias={hasBias}) via coopmat2: maxRelErr={maxRel:P2}");
            Assert.True(maxRel < 0.02f, $"CoopMat2 Linear (M={M},K={K},N={N},bias={hasBias}) maxRelErr {maxRel:P2} at [{firstM},{firstN}] too high.");
        }
        finally { input.Dispose(); weight.Dispose(); bias?.Dispose(); output.Dispose(); }
    }

    /// <summary>Even with the opt-in enabled, <c>MatMul</c>/<c>BatchedMatMul</c> (transposeB=false — the
    /// combination <see cref="VulkanBackend.TryDispatchCoopMat2"/> doesn't support) must fall through to
    /// coopmat1/tiled cleanly rather than throwing — <c>DispatchMatmul</c>'s call-site gate
    /// (<c>!transposeA &amp;&amp; transposeB</c>) exists specifically for this.</summary>
    [Fact]
    public void Backend_MatMul_CoopMat2OptIn_FallsThroughForUnsupportedTranspose()
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16 || !backend.Vk.HasCooperativeMatrix2)
        {
            _out.WriteLine("SKIPPED: no F16/coopmat2 support");
            return;
        }
        backend.EnableCoopMat2 = true;

        const int M = 64, K = 64, N = 64;
        Tensor a = new(new TensorShape(M, K), DType.F16);
        Tensor b = new(new TensorShape(K, N), DType.F16);
        Tensor output = new(new TensorShape(M, N), DType.F16);
        try
        {
            Random rng = new(55);
            Span<Half> aS = a.AsSpan<Half>();
            Span<Half> bS = b.AsSpan<Half>();
            for (int i = 0; i < M * K; i++) aS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);
            for (int i = 0; i < K * N; i++) bS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.1f);

            (long coopMat2Before, _, _) = backend.GemmEngagementCounts;
            backend.MatMul(output, a, b);   // transposeA=false, transposeB=false — not exercised by TryDispatchCoopMat2
            (long coopMat2After, _, _) = backend.GemmEngagementCounts;

            Assert.Equal(coopMat2Before, coopMat2After);   // must NOT have gone through coopmat2

            ReadOnlySpan<Half> oS = output.AsReadOnlySpan<Half>();
            float maxRel = 0f;
            for (int m = 0; m < M; m++)
                for (int n = 0; n < N; n++)
                {
                    float acc = 0f;
                    for (int k = 0; k < K; k++) acc += (float)aS[m * K + k] * (float)bS[k * N + n];
                    float rel = MathF.Abs((float)oS[m * N + n] - acc) / MathF.Max(1e-3f, MathF.Abs(acc));
                    if (rel > maxRel) maxRel = rel;
                }
            Assert.True(maxRel < 0.02f, $"MatMul with coopmat2 opted in maxRelErr {maxRel:P2} too high.");
        }
        finally { a.Dispose(); b.Dispose(); output.Dispose(); }
    }

    /// <summary>Same residency hazard class as <c>VulkanInt8GemmTests.Backend_Linear_Int8OptIn_
    /// BiasSurvivesDownstreamGpuConsumption</c>: coopmat2's bias is applied via a follow-up
    /// <see cref="VulkanBackend.BroadcastAdd"/> dispatch (not fused into the shader), so this proves that
    /// dispatch's result — not a stale pre-bias buffer — is what a downstream GPU consumer with no
    /// intervening host read actually sees.</summary>
    [Fact]
    public void Backend_Linear_CoopMat2OptIn_BiasSurvivesDownstreamGpuConsumption()
    {
        if (!VulkanAvailable()) { _out.WriteLine("SKIPPED: no Vulkan device"); return; }
        using VulkanBackend backend = new();
        if (!backend.Capabilities.SupportsF16 || !backend.Vk.HasCooperativeMatrix2)
        {
            _out.WriteLine("SKIPPED: no F16/coopmat2 support");
            return;
        }
        backend.EnableCoopMat2 = true;

        const int M = 33, K = 64, N = 32;   // M not 16-aligned — forces coopmat2, not coopmat1
        Tensor input = new(new TensorShape(M, K), DType.F16);
        Tensor weight = new(new TensorShape(N, K), DType.F16);
        Tensor bias = new(new TensorShape(N), DType.F16);
        Tensor linearOut = new(new TensorShape(M, N), DType.F16);
        Tensor siluOut = new(new TensorShape(M, N), DType.F16);
        try
        {
            Random rng = new(21);
            Span<Half> iS = input.AsSpan<Half>();
            Span<Half> wS = weight.AsSpan<Half>();
            Span<Half> bS = bias.AsSpan<Half>();
            for (int i = 0; i < M * K; i++) iS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.05f);
            for (int i = 0; i < N * K; i++) wS[i] = (Half)((float)(rng.NextDouble() * 2 - 1) * 0.05f);
            // Large bias relative to dot-product magnitude: if the downstream Silu sees the pre-bias
            // value instead, the mismatch is unmistakable.
            for (int n = 0; n < N; n++) bS[n] = (Half)(5.0f + n * 0.1f);

            (long coopMat2Before, _, _) = backend.GemmEngagementCounts;
            backend.Linear(linearOut, input, weight, bias);       // caches linearOut as GPU-resident
            (long coopMat2After, _, _) = backend.GemmEngagementCounts;
            Assert.Equal(coopMat2Before + 1, coopMat2After);
            backend.Silu(siluOut, linearOut);                     // consumes it with NO intervening host read

            ReadOnlySpan<Half> siluS = siluOut.AsReadOnlySpan<Half>();
            for (int m = 0; m < M; m++)
                for (int n = 0; n < N; n++)
                {
                    double acc = (float)bS[n];
                    for (int k = 0; k < K; k++) acc += (double)(float)iS[m * K + k] * (float)wS[n * K + k];
                    double expectedSilu = acc / (1.0 + Math.Exp(-acc));
                    Assert.True(Math.Abs((float)siluS[m * N + n] - expectedSilu) < Math.Abs(expectedSilu) * 0.05 + 0.05,
                        $"Silu(Linear(...))[{m},{n}]={(float)siluS[m * N + n]:G6} vs expected={expectedSilu:G6} — " +
                        "downstream GPU consumer likely saw a stale pre-bias buffer.");
                }
        }
        finally { input.Dispose(); weight.Dispose(); bias.Dispose(); linearOut.Dispose(); siluOut.Dispose(); }
    }
}
