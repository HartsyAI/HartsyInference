using System.Runtime.InteropServices;
using HartsyInference.Core.Tensors;
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

    private static Tensor Filled(float value)
    {
        Tensor tensor = new(new TensorShape(Length), DType.F32);
        float* data = (float*)tensor.DataPointer;
        for (int i = 0; i < Length; i++)
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
