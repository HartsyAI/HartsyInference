using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Whether weights preloaded on CUDA actually leave the device when released, and what the pool trim returns. Uses synthetic
/// tensors only, so it runs on any CUDA card with room for 256 MiB. It answers the residency question behind audit B3 without model weights.</summary>
[Collection(DeviceMemoryCollection.Name)]
public sealed class Fibo2ResidencyGpuTests(ITestOutputHelper output)
{
    private const long TensorBytes = 32L * 1024 * 1024;
    private const long ToleranceBytes = 64L * 1024 * 1024;

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void FreeWeights_LeavesTheDevice_AndTheTrimReturnsTheFreedMemory()
    {
        if (!CudaContext.IsAvailable())
        {
            output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaBackend backend = new(0, PtxDirectory());
        Tensor[] weights = Enumerable.Range(0, 8).Select(_ => new Tensor(new TensorShape(TensorBytes / 2), DType.F16)).ToArray();
        try
        {
            DeviceMemoryUsage before = backend.GetDeviceMemoryUsage();
            backend.PreloadWeights(weights);
            Assert.All(weights, w => Assert.True(backend.IsWeightResident(w), "preloaded weight not resident"));
            DeviceMemoryUsage loaded = backend.GetDeviceMemoryUsage();

            backend.FreeWeights(weights);
            Assert.All(weights, w => Assert.False(backend.IsWeightResident(w), "freed weight still resident"));
            DeviceMemoryUsage freed = backend.GetDeviceMemoryUsage();

            backend.TrimMemoryPool();
            DeviceMemoryUsage trimmed = backend.GetDeviceMemoryUsage();

            output.WriteLine($"free MiB: before {before.FreeBytes >> 20}, loaded {loaded.FreeBytes >> 20}, freed {freed.FreeBytes >> 20}, trimmed {trimmed.FreeBytes >> 20}");
            output.WriteLine($"pool MiB: before reserved {before.PoolReservedBytes >> 20} used {before.PoolUsedBytes >> 20}; freed reserved {freed.PoolReservedBytes >> 20} used {freed.PoolUsedBytes >> 20}; trimmed reserved {trimmed.PoolReservedBytes >> 20} used {trimmed.PoolUsedBytes >> 20}");

            // The preload must have taken device memory, and the free must give it back.
            Assert.True(loaded.FreeBytes < before.FreeBytes - TensorBytes * weights.Length / 2, "preload did not take device memory");
            Assert.True(freed.FreeBytes >= before.FreeBytes - ToleranceBytes, "freed weights did not return to the device");
        }
        finally
        {
            foreach (Tensor weight in weights)
                weight.Dispose();
        }
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        throw new DirectoryNotFoundException("HartsyInference.Cuda/Ptx was not found above the test binary.");
    }
}
