using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>What <see cref="CudaBackend.CopyTo"/> accepts as a CUDA destination: a tensor over a device pointer. A tensor
/// merely constructed with a CUDA device holds a host buffer, which the driver must never be handed as a device pointer.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class CudaCopyToDestinationGpuTests
{
    private const int Elements = 256;

    [CudaFact]
    public void CudaDestinationWithoutADeviceBuffer_IsRefused()
    {
        using CudaBackend backend = new(0, PtxDirectory());
        using Tensor host = Ramp();
        using Tensor labelled = new(host.Shape, DType.F32, backend.Device);

        ArgumentException refused = Assert.Throws<ArgumentException>(() => backend.CopyTo(labelled, host));
        Assert.Equal("destination", refused.ParamName);
        Assert.Contains("no device buffer", refused.Message);

        // A view shares its owner's host buffer, so it is refused as its owner is.
        using Tensor view = labelled.Reshape(new TensorShape(Elements / 2, 2));
        Assert.Throws<ArgumentException>(() => backend.CopyTo(view, host));
    }

    [CudaFact]
    public unsafe void CudaDestinationOverADevicePointer_CopiesEveryDirection()
    {
        using CudaBackend backend = new(0, PtxDirectory());
        using Tensor host = Ramp();
        using Tensor back = new(host.Shape, DType.F32);
        ulong first = 0, second = 0;
        try
        {
            first = CudaMemory.AllocatePersistent(Elements * sizeof(float));
            second = CudaMemory.AllocatePersistent(Elements * sizeof(float));
            using Tensor uploaded = new((void*)first, host.Shape, DType.F32, backend.Device);
            using Tensor copied = new((void*)second, host.Shape, DType.F32, backend.Device);

            backend.CopyTo(uploaded, host);
            backend.CopyTo(copied, uploaded);
            backend.CopyTo(back, copied);

            Assert.Equal(host.AsReadOnlySpan<float>().ToArray(), back.AsReadOnlySpan<float>().ToArray());
        }
        finally
        {
            CudaMemory.Free(first);
            CudaMemory.Free(second);
        }
    }

    private static Tensor Ramp()
    {
        Tensor tensor = new(new TensorShape(Elements), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = i + 0.5f;
        return tensor;
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate))
                return candidate;
        }
        throw new DirectoryNotFoundException("Could not find src/HartsyInference.Cuda/Ptx above the test binary.");
    }

    /// <summary>A fact reported as skipped, never passed, when no CUDA device opens.</summary>
    private sealed class CudaFactAttribute : FactAttribute
    {
        public CudaFactAttribute()
        {
            if (!CudaContext.IsAvailable()) Skip = "SKIPPED: CUDA unavailable.";
        }
    }
}
