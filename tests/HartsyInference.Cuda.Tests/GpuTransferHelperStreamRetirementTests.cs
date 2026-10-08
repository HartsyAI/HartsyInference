using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>Regression coverage for the LLMAssistant back-to-back-turn crash: <c>GpuTransferHelper.UploadTo</c>
/// read <c>State.StreamHandle</c> directly, and <see cref="GpuTransferHelper.CompleteRetire"/> zeroes that field on
/// backend disposal. A caller that resolved its <c>State</c> earlier in one op (<c>CopyToDevice</c> at the top of
/// an RmsNorm call, say) and only reads <c>StreamHandle</c> later in the SAME op — right before the native
/// <c>cuMemcpyHtoDAsync</c> — could observe the zero if disposal raced it from another thread: a bare
/// <c>nint StreamHandle = 0</c> is a legal "use the legacy default stream" argument to the driver, not a disposed-
/// object signal, and a legacy-stream copy into a buffer the stream-ordered pool allocated elsewhere is exactly
/// the shape of the intermittent <c>CUDA_ERROR_INVALID_VALUE</c> the real incident logged (journal
/// 2026-10-01T19:58:54Z, SwarmUI's LLMAssistant extension, <c>TextService.RunText</c> -&gt;
/// <c>GenericTransformer.Layer.Forward</c> -&gt; <c>CudaBackend.RmsNorm</c> -&gt; <c>GpuTransferHelper.UploadTo</c>).
///
/// <para>This reproduces the race's OBSERVABLE EFFECT deterministically without timing two real requests against
/// each other: <see cref="GpuTransferHelper.State.StreamHandle"/> is a plain mutable field, so setting it to zero
/// by hand is indistinguishable, from <c>UploadTo</c>'s point of view, from a concurrent <c>CompleteRetire</c>
/// landing between this test's <c>Resolve()</c> and its native call. <see cref="ModelSwapSoakTests"/> covers the
/// sibling finalizer-callback race (<c>State.Unregistered</c>'s own reason for existing); this covers the ordinary,
/// non-finalizer op path that had no equivalent guard.</para></summary>
[Collection("CudaSerial")]
public sealed class GpuTransferHelperStreamRetirementTests
{
    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(dir) ? dir : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }

    private readonly ITestOutputHelper _output;

    public GpuTransferHelperStreamRetirementTests(ITestOutputHelper output) => _output = output;

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void UploadAgainstAZeroedStreamHandle_ThrowsObjectDisposed_NotARawCudaInvalidValue()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine($"SKIPPED: CUDA unavailable: {CudaContext.LastUnavailableReason}");
            return;
        }

        // Ordinal 1 (not the default 0): keeps this test off whichever card is fastest-first on a given box, so
        // it never contends with a concurrent timed benchmark that (by this repo's own convention) prefers
        // ordinal 0. The mechanism under test does not depend on which physical device it runs against.
        if (CudaContext.GetDeviceCount() < 2)
        {
            _output.WriteLine("SKIPPED: needs a second CUDA device (ordinal 1).");
            return;
        }
        using CudaBackend backend = new(1, PtxDir());
        // Binds this thread's ambient to `backend`'s State exactly as every real op does via EnterOp — the
        // static GpuTransferHelper.CopyToDevice call below has no op scope of its own to do it.
        backend.BindAmbient();
        GpuTransferHelper.State state = backend.TransferState;
        nint originalStream = state.StreamHandle;
        Assert.NotEqual(0, originalStream);

        using Tensor weight = new(new TensorShape(8), DType.F32);
        Span<float> values = weight.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = i;

        // Simulates CompleteRetire landing between a caller's Resolve() and its native upload call — the exact
        // gap UploadTo had. Restored in `finally` so the backend's own Dispose() still tears down cleanly.
        state.StreamHandle = 0;
        try
        {
            ObjectDisposedException thrown = Assert.Throws<ObjectDisposedException>(
                () => GpuTransferHelper.CopyToDevice(weight));
            Assert.Contains("retired", thrown.Message, StringComparison.OrdinalIgnoreCase);
            _output.WriteLine($"Correctly threw: {thrown.Message}");
        }
        finally
        {
            state.StreamHandle = originalStream;
        }

        // The guard must not have left anything (a transient allocation, a half-published cache entry) behind:
        // the SAME tensor uploads cleanly now that the stream is live again.
        ulong devicePtr = GpuTransferHelper.CopyToDevice(weight);
        Assert.NotEqual(0UL, devicePtr);
    }
}
