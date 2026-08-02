using HartsyInference.Core.Logging;

namespace HartsyInference.Cuda;

/// <summary>One CUDA device's placement-relevant facts, as probed by <see cref="CudaTopology.Probe"/>.</summary>
/// <param name="Ordinal">CUDA enumeration ordinal (fastest-first by default — NOT nvidia-smi order).</param>
/// <param name="Name">Device name, e.g. "NVIDIA GeForce RTX 4090".</param>
/// <param name="TotalMemoryBytes">Total device memory.</param>
/// <param name="FreeMemoryBytes">Free device memory at probe time (other processes included).</param>
/// <param name="CcMajor">Compute capability major.</param>
/// <param name="CcMinor">Compute capability minor.</param>
public readonly record struct GpuTopologyInfo(
    int Ordinal, string Name, long TotalMemoryBytes, long FreeMemoryBytes, int CcMajor, int CcMinor);

/// <summary>Device-topology probe for the placement planner: per-device VRAM/compute-capability now, peer-access
/// capability once the peer-copy bindings land. Read-only queries; the free-VRAM number binds each device's
/// primary context briefly.</summary>
public static class CudaTopology
{
    /// <summary>Probes every visible CUDA device. Empty when CUDA is unavailable.</summary>
    public static IReadOnlyList<GpuTopologyInfo> Probe()
    {
        List<GpuTopologyInfo> devices = [];
        if (!CudaContext.IsAvailable())
        {
            return devices;
        }
        int count = CudaContext.GetDeviceCount();
        for (int ordinal = 0; ordinal < count; ordinal++)
        {
            try
            {
                using CudaContext context = new CudaContext(ordinal);
                (nuint free, nuint total) = context.GetMemoryInfo();
                devices.Add(new GpuTopologyInfo(
                    ordinal,
                    context.DeviceName,
                    (long)total,
                    (long)free,
                    context.ComputeCapabilityMajor,
                    context.ComputeCapabilityMinor));
            }
            catch (Exception ex)
            {
                Logs.Warning($"[Cuda] Topology probe failed for device {ordinal}: {ex.Message}");
            }
        }
        return devices;
    }
}
