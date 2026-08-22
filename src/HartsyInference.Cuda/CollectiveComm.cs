using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;

namespace HartsyInference.Cuda;

/// <summary>Factory for <see cref="ICollectiveComm"/>: NCCL when every rank is a distinct-device <see cref="CudaBackend"/> and libnccl resolves, else the host-staged fallback. Never throws for a missing library — the transport decision is logged either way (greppable <c>[Collective]</c>).</summary>
public static class CollectiveComm
{
    /// <summary>Creates the best available communicator over one backend per rank.</summary>
    public static ICollectiveComm Create(IReadOnlyList<IBackend> backends)
    {
        ArgumentNullException.ThrowIfNull(backends);
        if (backends.Count < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(backends), backends.Count, "A collective needs at least 2 ranks.");
        }
        string? fallbackReason = null;
        List<CudaBackend> cuda = new(backends.Count);
        foreach (IBackend backend in backends)
        {
            if (backend is CudaBackend cudaBackend)
            {
                cuda.Add(cudaBackend);
            }
            else
            {
                fallbackReason = $"rank backend {backend.GetType().Name} is not CUDA";
                break;
            }
        }
        if (fallbackReason is null && cuda.Select(b => b.Device.Ordinal).Distinct().Count() != cuda.Count)
        {
            fallbackReason = "duplicate CUDA ordinals (NCCL needs one distinct device per rank)";
        }
        if (fallbackReason is null && !NcclApi.IsLoadable)
        {
            fallbackReason = "libnccl not resolvable (set HARTSY_NCCL_DIR or place libnccl.so.2 in a probe dir)";
        }
        if (fallbackReason is null)
        {
            try
            {
                NcclComm comm = new(cuda);
                int v = NcclApi.Version;
                Logs.Info($"[Collective] transport=nccl version={v / 10000}.{v % 10000 / 100}.{v % 100} ranks={backends.Count}");
                return comm;
            }
            catch (Exception ex)
            {
                fallbackReason = $"NCCL init failed: {ex.Message}";
            }
        }
        Logs.Info($"[Collective] transport=host-staged ranks={backends.Count} ({fallbackReason})");
        return new HostStagedComm(backends.Count);
    }
}
