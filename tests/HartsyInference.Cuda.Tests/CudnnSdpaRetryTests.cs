using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Correctness gate for the cuDNN SDPA host-RAM-failure classify/retry/backoff hardening (see
/// docs plan notes for context: a genuinely transient host-allocation failure —
/// <c>CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED</c> — used to permanently disable the fused
/// attention path for a head dim forever, with no distinction from a genuinely structural incompatibility).
/// <see cref="CudnnStatusException.IsPermanent"/> classification is pure logic (no GPU needed, runs
/// everywhere); the retry/backoff/diagnostics behavior needs <see cref="CudaBackend.TestCudnnSdpaFaultInjector"/>
/// and so is gated on CUDA availability like every other GPU-touching test in this file's siblings.</summary>
[Collection("CudaSerial")]
public sealed unsafe class CudnnSdpaRetryTests
{
    private readonly ITestOutputHelper _output;
    public CudnnSdpaRetryTests(ITestOutputHelper output) => _output = output;

    private static uint _rng = 0x51ED270Bu;
    private static float Rand() { _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5; return (_rng & 0xFFFF) / 65535f - 0.5f; }
    private static Tensor Rnd(int a, int b, int c, int d) { Tensor t = new(new TensorShape(a, b, c, d), DType.F32); float* p = (float*)t.DataPointer; for (long i = 0; i < t.ElementCount; i++) p[i] = Rand(); return t; }


    // CudnnStatusException's constructor calls CudnnApi.ErrorString (a cuDNN P/Invoke) to format its
    // message — resolving "cudnn" to the actual library path is normally done lazily by CudaContext, which
    // this pure-logic test never touches. Register the resolver directly (safe without a live GPU device —
    // it's just a NativeLibrary.SetDllImportResolver hook) so the constructor can actually load the library.
    static CudnnSdpaRetryTests() => CudaLibraryResolver.Register();

    [Theory]
    [InlineData(2000, true)]   // BAD_PARAM category — structural, never retry
    [InlineData(2001, true)]
    [InlineData(3000, true)]   // NOT_SUPPORTED category — structural (subsumes e.g. arch mismatch)
    [InlineData(3007, true)]
    [InlineData(4000, false)]  // INTERNAL_ERROR category
    [InlineData(4003, false)]  // the exact status that motivated this work (HOST_ALLOCATION_FAILED)
    [InlineData(5000, false)]  // EXECUTION_FAILED category
    [InlineData(1001, false)]  // uncategorized (1000s) — defaults to transient, see IsPermanent's doc
    [InlineData(9999, false)]  // unrecognized/future category — must default to transient, not permanent
    public void IsPermanent_ClassifiesByCudnnStatusCategory(int status, bool expectedPermanent)
    {
        CudnnStatusException ex = new(status, "test op");
        Assert.Equal(expectedPermanent, ex.IsPermanent);
    }

    [Fact]
    public void Message_IncludesStatusAndWhat()
    {
        CudnnStatusException ex = new(4003, "softmax op create");
        Assert.Contains("softmax op create", ex.Message);
        Assert.Equal(4003, ex.Status);
    }

    [Fact]
    public void TransientFailure_BacksOffThenRecovers()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = System.IO.Path.Combine(AppContext.BaseDirectory, "Ptx");

        const int d = 64, heads = 4, s = 32;
        float scale = 1f / MathF.Sqrt(d);
        string? prevEnv = Environment.GetEnvironmentVariable("HARTSY_SDPA_CUDNN");
        Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", "1");
        try
        {
            using CudaBackend backend = new(0, ptxDir);
            using Tensor q = Rnd(1, heads, s, d);
            using Tensor k = Rnd(1, heads, s, d);
            using Tensor v = Rnd(1, heads, s, d);
            using Tensor outT = new(new TensorShape(1, heads, s, d), DType.F32);

            int injections = 0;
            backend.TestCudnnSdpaFaultInjector = dim =>
            {
                injections++;
                return new CudnnStatusException(4003, "injected transient failure"); // INTERNAL_ERROR category
            };

            backend.ScaledDotProductAttention(outT, q, k, v, null, scale, allowF16: true);
            Assert.Equal(1, injections);
            Assert.False(backend.CudnnSdpaEngaged);
            Assert.True(backend.CudnnSdpaDimDiagnostics.TryGetValue(d, out (int ConsecutiveFailures, bool Permanent, DateTimeOffset? NextRetryAt) diag1));
            Assert.False(diag1.Permanent);
            Assert.Equal(1, diag1.ConsecutiveFailures);
            _output.WriteLine($"after 1 injected failure: {diag1}");

            // Still inside the backoff window — the dispatch gate must reject the dim BEFORE TryCudnnSdpa
            // (and therefore the injector) is even reached again.
            backend.ScaledDotProductAttention(outT, q, k, v, null, scale, allowF16: true);
            Assert.Equal(1, injections);

            // Base backoff for the first failure is 2000ms (CudnnSdpaBackoffMs) — wait past it for real,
            // proving actual elapsed-time eligibility rather than mocking the clock. Keep the injector
            // armed for this attempt too (rather than clearing it and hoping a REAL cuDNN call succeeds):
            // this box's host RAM has been genuinely tight all session, so a real retry succeeding isn't
            // guaranteed on any given run — asserting on that would make this test flaky for reasons
            // outside the code under test. What's deterministic and worth asserting is the MECHANISM: the
            // gate allows a fresh attempt once the backoff window elapses (injector invoked a 2nd time,
            // proving it wasn't gated), and the failure count/backoff grow accordingly.
            System.Threading.Thread.Sleep(2200);
            backend.ScaledDotProductAttention(outT, q, k, v, null, scale, allowF16: true);
            Assert.Equal(2, injections);
            Assert.True(backend.CudnnSdpaDimDiagnostics.TryGetValue(d, out (int ConsecutiveFailures, bool Permanent, DateTimeOffset? NextRetryAt) diag2));
            Assert.False(diag2.Permanent);
            Assert.Equal(2, diag2.ConsecutiveFailures);
            _output.WriteLine($"after backoff window elapsed, retried for real (injector invoked again): {diag2}");

            // NOW let a genuinely uninjected attempt through — best-effort signal only (not asserted) since
            // whether cuDNN actually succeeds depends on this box's real, currently-uncontrolled RAM
            // pressure, which is exactly the condition this whole mechanism exists to tolerate gracefully
            // rather than guarantee away.
            System.Threading.Thread.Sleep(4200); // 2nd failure's backoff is 4000ms (2000 * 2^1)
            backend.TestCudnnSdpaFaultInjector = null;
            backend.ScaledDotProductAttention(outT, q, k, v, null, scale, allowF16: true);
            _output.WriteLine(backend.CudnnSdpaEngaged
                ? "bonus: a real (uninjected) retry also succeeded on this box"
                : "a real (uninjected) retry did not succeed on this box right now (real host RAM pressure) — not a test failure, this is exactly the condition the retry/backoff mechanism is designed to tolerate");
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", prevEnv);
        }
    }

    [Fact]
    public void PermanentFailure_NeverRetries()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        string ptxDir = System.IO.Path.Combine(AppContext.BaseDirectory, "Ptx");

        const int d = 128, heads = 4, s = 32;
        float scale = 1f / MathF.Sqrt(d);
        string? prevEnv = Environment.GetEnvironmentVariable("HARTSY_SDPA_CUDNN");
        Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", "1");
        try
        {
            using CudaBackend backend = new(0, ptxDir);
            using Tensor q = Rnd(1, heads, s, d);
            using Tensor k = Rnd(1, heads, s, d);
            using Tensor v = Rnd(1, heads, s, d);
            using Tensor outT = new(new TensorShape(1, heads, s, d), DType.F32);

            int injections = 0;
            backend.TestCudnnSdpaFaultInjector = dim =>
            {
                injections++;
                return new CudnnStatusException(3000, "injected structural failure"); // NOT_SUPPORTED category
            };

            backend.ScaledDotProductAttention(outT, q, k, v, null, scale, allowF16: true);
            Assert.Equal(1, injections);
            Assert.True(backend.CudnnSdpaDimDiagnostics.TryGetValue(d, out (int ConsecutiveFailures, bool Permanent, DateTimeOffset? NextRetryAt) diag));
            Assert.True(diag.Permanent);

            // Immediately eligible-looking (no backoff window to wait out for a permanent failure) — must
            // still never invoke the injector again.
            backend.ScaledDotProductAttention(outT, q, k, v, null, scale, allowF16: true);
            Assert.Equal(1, injections);
            Assert.False(backend.CudnnSdpaEngaged);
        }
        finally
        {
            Environment.SetEnvironmentVariable("HARTSY_SDPA_CUDNN", prevEnv);
        }
    }
}
