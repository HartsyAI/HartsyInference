using System.Runtime.InteropServices;
using HartsyInference.Core.Tensors;
using HartsyInference.Gpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>An op must run in its own backend's context whatever the calling thread was left bound to. The driver's
/// current context is per thread and process-wide, but the binding <see cref="CudaContext.EnsureCurrent"/> remembers
/// is per copy of this assembly: SwarmUI loads one copy per extension (AudioLab, LLMAssistant, the image backend),
/// all sharing the thread pool, and any other CUDA library can rebind a thread too. These tests change the binding
/// behind the backend's back the way such a neighbour does, then run an op.</summary>
[Collection("CudaSerial")]
[Trait("Category", "GpuIntegration")]
public sealed unsafe class CudaContextRebindTests
{
    private const int Length = 4096;

    /// <summary>1 MiB of F32: the smallest tensor the cache will promote to a resident weight on its own.</summary>
    private const int PromotableLength = 1 << 18;

    private readonly ITestOutputHelper _output;

    public CudaContextRebindTests(ITestOutputHelper output) => _output = output;

    /// <summary>A neighbour that leaves its own context bound. A resident weight is allocated in whatever context is
    /// current, so one preloaded after the neighbour bound its context belonged to the neighbour's (on another
    /// device, another card's memory, which the backend's kernels then fault on) and died with it.</summary>
    [Fact]
    public void WeightPreloadedAfterAnotherContextWasBound_BelongsToTheBackendsContext()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine($"SKIPPED: CUDA unavailable ({CudaContext.LastUnavailableReason}).");
            return;
        }
        using CudaBackend backend = new(0, PtxDir());
        using Tensor weight = Filled(0.5f);
        nint neighbour = 0;
        try
        {
            AssertAddIsCorrect(backend);

            neighbour = CreateContextOnSameDevice(backend.Context.DeviceOrdinal);
            backend.PreloadWeights([weight]);

            Assert.Equal(backend.Context.Handle, OwningContext(backend.TransferState.WeightCache[weight]));
            AssertAddIsCorrect(backend);
        }
        finally
        {
            backend.FreeWeights([weight]);
            if (neighbour != 0)
            {
                DestroyContext(neighbour);
            }
            backend.Context.MakeCurrent();
        }
    }

    /// <summary>The production shape: a neighbour leaves ANOTHER device's context bound, so the op's allocations,
    /// copies and kernels, and the free-VRAM reading the audio switch check makes, would all land on the wrong card.
    /// Opt-in (<c>HARTSY_CUDA_REBIND_TWO_DEVICES=1</c>) because it creates a context on a second GPU.</summary>
    [Fact]
    public void OpAfterAnotherDevicesContextWasBound_RunsOnItsOwnDevice()
    {
        if (Environment.GetEnvironmentVariable("HARTSY_CUDA_REBIND_TWO_DEVICES") != "1")
        {
            _output.WriteLine("SKIPPED: set HARTSY_CUDA_REBIND_TWO_DEVICES=1 to run (needs two visible GPUs).");
            return;
        }
        if (!CudaContext.IsAvailable() || CudaContext.GetDeviceCount() < 2)
        {
            _output.WriteLine("SKIPPED: needs CUDA with two visible devices.");
            return;
        }
        using CudaBackend first = new(0, PtxDir());
        using CudaBackend second = new(1, PtxDir());
        try
        {
            long firstTotal = first.GetVramInfo().TotalBytes;
            AssertAddIsCorrect(first);

            CudaDriverApi.cuCtxSetCurrent(second.Context.Handle).ThrowOnError();

            Assert.Equal(firstTotal, first.GetVramInfo().TotalBytes);
            AssertAddIsCorrect(first);
        }
        finally
        {
            first.Context.MakeCurrent();
        }
    }

    /// <summary>The same neighbour, arriving at a FREE instead of an op: a tensor is dropped on a thread whose current
    /// context is another device's. A synchronous free resolves against the current context rather than the buffer's,
    /// so each callback that frees must make its own backend's context current first — the one for an activation, and
    /// the one for a weight the cache promoted on its own, which is the buffer that takes the synchronous path. Every
    /// tensor here is disposed after the thread has been rebound to the other device. Opt-in like the op test, for
    /// the same reason.</summary>
    [Fact]
    public void TensorsDisposedAfterAnotherDevicesContextWasBound_FreeIntoTheirOwnDevice()
    {
        if (Environment.GetEnvironmentVariable("HARTSY_CUDA_REBIND_TWO_DEVICES") != "1")
        {
            _output.WriteLine("SKIPPED: set HARTSY_CUDA_REBIND_TWO_DEVICES=1 to run (needs two visible GPUs).");
            return;
        }
        if (!CudaContext.IsAvailable() || CudaContext.GetDeviceCount() < 2)
        {
            _output.WriteLine("SKIPPED: needs CUDA with two visible devices.");
            return;
        }
        using CudaBackend first = new(0, PtxDir());
        using CudaBackend second = new(1, PtxDir());
        // The cache promotes only when it would leave its free-VRAM headroom (1536 MB by default) untouched.
        if (first.GetVramInfo().FreeBytes < (1536L + 64L) << 20)
        {
            _output.WriteLine("SKIPPED: not enough free VRAM on device 0 for the cache to promote a weight.");
            return;
        }
        GpuTransferHelper.State state = first.TransferState;
        using Tensor a = Filled(1.5f, PromotableLength);
        using Tensor b = Filled(2.25f, PromotableLength);
        using Tensor sum = new(new TensorShape(PromotableLength), DType.F32);
        try
        {
            // The first sight of each input is a transient upload; a second, with the host data unchanged, promotes it.
            first.Add(sum, a, b);
            first.Add(sum, a, b);
            Assert.Equal(GpuResidencyTier.Weight, state.TierOf(a));
            Assert.Equal(GpuResidencyTier.Weight, state.TierOf(b));
            Assert.Equal(GpuResidencyTier.Activation, state.TierOf(sum));
            Assert.Equal(2, state.PersistentBuffers.Count);

            // Rebound before each release, so every free arrives on a thread that last ran on the other device.
            CudaDriverApi.cuCtxSetCurrent(second.Context.Handle).ThrowOnError();
            sum.Dispose();
            CudaDriverApi.cuCtxSetCurrent(second.Context.Handle).ThrowOnError();
            a.Dispose();
            CudaDriverApi.cuCtxSetCurrent(second.Context.Handle).ThrowOnError();
            b.Dispose();

            Assert.Equal(0, state.ActivationCount);
            Assert.Equal(0, state.WeightCount);
            Assert.Equal(0, state.CachedBufferCount);
            Assert.Empty(state.PersistentBuffers);
            AssertAddIsCorrect(first);
            AssertAddIsCorrect(second);
        }
        finally
        {
            first.Context.MakeCurrent();
        }
    }

    /// <summary>A second, non-primary context on <paramref name="ordinal"/>, left current on this thread the way a
    /// neighbour leaves its own. The engine binds no context-creation entry point on purpose, so the test resolves it.</summary>
    private static nint CreateContextOnSameDevice(int ordinal)
    {
        CudaDriverApi.cuDeviceGet(out int device, ordinal).ThrowOnError();
        delegate* unmanaged<nint*, uint, int, int> create =
            (delegate* unmanaged<nint*, uint, int, int>)NativeLibrary.GetExport(Driver(), "cuCtxCreate_v2");
        nint context;
        create(&context, 0, device).ThrowOnError();
        return context;
    }

    private static void DestroyContext(nint context)
    {
        delegate* unmanaged<nint, int> destroy = (delegate* unmanaged<nint, int>)NativeLibrary.GetExport(Driver(), "cuCtxDestroy_v2");
        destroy(context).ThrowOnError();
    }

    /// <summary>The context a device allocation belongs to (<c>CU_POINTER_ATTRIBUTE_CONTEXT</c>).</summary>
    private static nint OwningContext(ulong devicePointer)
    {
        delegate* unmanaged<void*, int, ulong, int> query =
            (delegate* unmanaged<void*, int, ulong, int>)NativeLibrary.GetExport(Driver(), "cuPointerGetAttribute");
        nint context;
        query(&context, 1, devicePointer).ThrowOnError();
        return context;
    }

    private static nint Driver() => NativeLibrary.Load(OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1");

    private static void AssertAddIsCorrect(CudaBackend backend)
    {
        using Tensor a = Filled(1.5f);
        using Tensor b = Filled(2.25f);
        using Tensor sum = new(new TensorShape(Length), DType.F32);
        backend.Add(sum, a, b);
        float* values = (float*)sum.DataPointer;
        for (int i = 0; i < Length; i++)
        {
            Assert.Equal(3.75f, values[i]);
        }
    }

    private static Tensor Filled(float value, int length = Length)
    {
        Tensor tensor = new(new TensorShape(length), DType.F32);
        float* data = (float*)tensor.DataPointer;
        for (int i = 0; i < length; i++)
        {
            data[i] = value;
        }
        return tensor;
    }

    private static string PtxDir()
    {
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(ptxDir)
            ? ptxDir
            : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }
}
