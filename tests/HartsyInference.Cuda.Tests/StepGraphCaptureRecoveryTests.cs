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
    public void CaptureInvalidatedByALegacyStreamCall_ResetsCleanAndTheBackendRecaptures()
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
            // A legacy-stream memset, as any caller outside the engine's transfer helpers might issue; it fails too.
            Record.Exception(() => CudaDriverApi.cuMemsetD8(scratch, 0, 4096).ThrowOnError());
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

    /// <summary>The engine's own synchronous transfers run on the calling backend's stream, so a second backend on the
    /// same GPU can upload, download, fill and allocate while the first captures, and both finish.</summary>
    [Fact]
    public void AnotherBackendsSynchronousTransfers_DuringACapture_NeitherFailNorInvalidateIt()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using Tensor input = RandomF32(new TensorShape(2, Dim), 831);
        using Tensor weight = RandomF32(new TensorShape(Dim, Dim), 832);
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
            ulong scratch = GpuTransferHelper.AllocateDevice(4 * sizeof(float));

            capturing.StepGraphBegin();
            capturing.Linear(output, input, weight, bias: null);
            GpuTransferHelper.SetAmbient(other.TransferState);
            float* host = stackalloc float[4] { 1f, 2f, 3f, 4f };
            float* back = stackalloc float[4];
            CudaMemory.CopyHostToDevice(scratch, host, 4 * sizeof(float));
            CudaMemory.Fill32(scratch, BitConverter.SingleToUInt32Bits(7f), 2);
            CudaMemory.CopyDeviceToHost(back, scratch, 4 * sizeof(float));
            ulong persistent = CudaMemory.Allocate(1 << 20);
            CudaMemory.Free(persistent);
            capturing.StepGraphEndAndLaunch();
            capturing.Sync();

            Assert.Equal([7f, 7f, 3f, 4f], new[] { back[0], back[1], back[2], back[3] });
            Assert.True(capturing.StepGraphReady);
            Assert.Equal(expected, ((float*)output.DataPointer)[0], 3);
            GpuTransferHelper.SetAmbient(other.TransferState);
            GpuTransferHelper.FreeDevice(scratch);
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
