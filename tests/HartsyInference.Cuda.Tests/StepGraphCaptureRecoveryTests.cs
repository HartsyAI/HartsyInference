using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Gpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>A step-graph capture that ends badly must cost one eager step, never the backend. A capture on the blocking
/// compute stream is invalidated by any use of the legacy stream in the same context, including another backend's
/// synchronous copy, and a host-wide memory release can arrive while one is open.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class StepGraphCaptureRecoveryTests
{
    private const int Dim = 64;
    private readonly ITestOutputHelper _output;

    public StepGraphCaptureRecoveryTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void CaptureInvalidatedByAnotherBackendsLegacyCopy_ResetsCleanAndTheBackendRecaptures()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using Tensor input = RandomF32(new TensorShape(2, Dim), 811);
        using Tensor weight = RandomF32(new TensorShape(Dim, Dim), 812);
        float expected = ExpectedFirst(input, weight);
        CudaBackend capturing = new(0, PtxDir());
        CudaBackend other = new(0, PtxDir());
        try
        {
            if (!((IBackend)capturing).StepGraphSupported) { _output.WriteLine("SKIPPED: StepGraph unsupported"); return; }
            capturing.HighPrecisionGemm = true;
            capturing.StepGraphOwner = this;
            capturing.PreloadWeights([weight]);
            using Tensor output = new(new TensorShape(2, Dim), DType.F32);
            GpuTransferHelper.SetAmbient(other.TransferState);
            ulong scratch = GpuTransferHelper.AllocateDevice(4096);

            capturing.StepGraphBegin();
            capturing.Linear(output, input, weight, bias: null);
            // The synchronous memset runs on the legacy stream; the offending call itself fails too.
            Record.Exception(() => CudaMemory.Zero(scratch, 4096));
            Assert.ThrowsAny<Exception>(() => capturing.Linear(output, input, weight, bias: null));

            capturing.StepGraphReset();
            GpuTransferHelper.SetAmbient(other.TransferState);
            GpuTransferHelper.FreeDevice(scratch);

            Assert.False(capturing.TransferState.TrackCaptureWindow);
            Assert.False(capturing.StepGraphReady);
            AssertCapturesAndReplays(capturing, input, weight, expected);
        }
        finally
        {
            DisposeBoth(capturing, other);
        }
    }

    [Fact]
    public void FreeAllDeviceMemory_WithACaptureOpen_ClosesItAndTheBackendRecaptures()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using Tensor input = RandomF32(new TensorShape(2, Dim), 821);
        using Tensor weight = RandomF32(new TensorShape(Dim, Dim), 822);
        float expected = ExpectedFirst(input, weight);
        CudaBackend backend = new(0, PtxDir());
        try
        {
            if (!((IBackend)backend).StepGraphSupported) { _output.WriteLine("SKIPPED: StepGraph unsupported"); return; }
            backend.HighPrecisionGemm = true;
            backend.StepGraphOwner = this;
            backend.PreloadWeights([weight]);
            using Tensor output = new(new TensorShape(2, Dim), DType.F32);

            backend.StepGraphBegin();
            backend.Linear(output, input, weight, bias: null);
            backend.FreeAllDeviceMemory();

            Assert.False(backend.TransferState.TrackCaptureWindow);
            Assert.False(backend.StepGraphReady);
            backend.StepGraphOwner = this;
            backend.PreloadWeights([weight]);
            AssertCapturesAndReplays(backend, input, weight, expected);
        }
        finally
        {
            DisposeBoth(backend, null);
        }
    }

    private static void AssertCapturesAndReplays(CudaBackend backend, Tensor input, Tensor weight, float expected)
    {
        using Tensor output = new(new TensorShape(2, Dim), DType.F32);
        backend.StepGraphBegin();
        backend.Linear(output, input, weight, bias: null);
        backend.StepGraphEndAndLaunch();
        backend.Sync();
        Assert.True(backend.StepGraphReady);
        Assert.Equal(expected, ((float*)output.DataPointer)[0], 3);
    }

    private void DisposeBoth(CudaBackend first, CudaBackend? second)
    {
        try { first.StepGraphReset(); } catch (Exception error) { _output.WriteLine($"teardown reset: {error.Message}"); }
        if (ReferenceEquals(first.StepGraphOwner, this)) first.StepGraphOwner = null;
        first.Dispose();
        second?.Dispose();
    }

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
            ptxDir = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return ptxDir;
    }

    private static Tensor RandomF32(TensorShape shape, int seed)
    {
        Tensor tensor = new(shape, DType.F32);
        Random rng = new(seed);
        float* p = (float*)tensor.DataPointer;
        for (long i = 0; i < tensor.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return tensor;
    }

    private static float ExpectedFirst(Tensor input, Tensor weight)
    {
        float* ip = (float*)input.DataPointer;
        float* wp = (float*)weight.DataPointer;
        double acc = 0;
        for (int k = 0; k < Dim; k++) acc += ip[k] * wp[k];
        return (float)acc;
    }
}
