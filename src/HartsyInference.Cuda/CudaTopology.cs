using HartsyInference.Core.Logging;

namespace HartsyInference.Cuda;

/// <summary>One CUDA device's placement-relevant facts, as probed by <see cref="CudaTopology.Probe"/>.</summary>
/// <param name="Ordinal">CUDA enumeration ordinal (fastest-first by default — NOT nvidia-smi order).</param>
/// <param name="Name">Device name, e.g. "NVIDIA GeForce RTX 4090".</param>
/// <param name="TotalMemoryBytes">Total device memory.</param>
/// <param name="FreeMemoryBytes">Free device memory at probe time (other processes included).</param>
/// <param name="CcMajor">Compute capability major.</param>
/// <param name="CcMinor">Compute capability minor.</param>
/// <param name="SmCount">Streaming-multiprocessor count — the per-device compute-throughput proxy.</param>
public readonly record struct GpuTopologyInfo(
    int Ordinal, string Name, long TotalMemoryBytes, long FreeMemoryBytes, int CcMajor, int CcMinor, int SmCount);

/// <summary>One DIRECTED device pair's peer-access facts, as probed by <see cref="CudaTopology.ProbeLinks"/>.</summary>
/// <param name="From">Accessing device ordinal.</param>
/// <param name="To">Peer device ordinal.</param>
/// <param name="PeerAccessSupported">True when From can map To's memory directly (PCIe P2P or NVLink).</param>
/// <param name="PerformanceRank">Driver's relative link ranking (lower is faster); -1 when no peer access.</param>
/// <param name="NativeAtomics">True when native atomic operations work across the link.</param>
/// <param name="LikelyNvLink">Heuristic: peer access + native atomics is the accepted driver-API signal for an NVLink-class link (NVML would be authoritative but is not a dependency).</param>
public readonly record struct GpuLinkInfo(
    int From, int To, bool PeerAccessSupported, int PerformanceRank, bool NativeAtomics, bool LikelyNvLink);

/// <summary>Device-topology probe for the placement planner: per-device VRAM/compute-capability/SM count plus the directed peer-link matrix. Read-only queries; the free-VRAM number binds each device's primary context briefly, and <see cref="ProbeLinks"/> never enables peer access.</summary>
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
                    context.ComputeCapabilityMinor,
                    context.MultiprocessorCount));
            }
            catch (Exception ex)
            {
                Logs.Warning($"[Cuda] Topology probe failed for device {ordinal}: {ex.Message}");
            }
        }
        return devices;
    }

    /// <summary>Probes every ordered device pair's peer-link capability. Query-only — peer access is never enabled, so context state is untouched. Empty when CUDA is unavailable or fewer than two devices.</summary>
    public static IReadOnlyList<GpuLinkInfo> ProbeLinks()
    {
        // TODO: per-link bandwidth microbench (needs GPU compute) — deferred, see ROADMAP §1 Phase 1.
        List<GpuLinkInfo> links = [];
        if (!CudaContext.IsAvailable())
        {
            return links;
        }
        int count = CudaContext.GetDeviceCount();
        for (int i = 0; i < count; i++)
        {
            for (int j = 0; j < count; j++)
            {
                if (i == j)
                {
                    continue;
                }
                try
                {
                    CudaDriverApi.cuDeviceGet(out int src, i).ThrowOnError();
                    CudaDriverApi.cuDeviceGet(out int dst, j).ThrowOnError();
                    CudaDriverApi.cuDeviceCanAccessPeer(out int can, src, dst).ThrowOnError();
                    int rank = -1;
                    bool access = false;
                    bool atomics = false;
                    if (can != 0)
                    {
                        CudaDriverApi.cuDeviceGetP2PAttribute(
                            out rank, CudaDriverApi.CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK, src, dst).ThrowOnError();
                        CudaDriverApi.cuDeviceGetP2PAttribute(
                            out int acc, CudaDriverApi.CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED, src, dst).ThrowOnError();
                        CudaDriverApi.cuDeviceGetP2PAttribute(
                            out int nat, CudaDriverApi.CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED, src, dst).ThrowOnError();
                        access = acc != 0;
                        atomics = nat != 0;
                    }
                    links.Add(new GpuLinkInfo(i, j, can != 0, rank, atomics, access && atomics));
                }
                catch (Exception ex)
                {
                    Logs.Warning($"[Cuda] Link probe {i}→{j} failed: {ex.Message}");
                    links.Add(new GpuLinkInfo(i, j, false, -1, false, false));
                }
            }
        }
        return links;
    }
}
