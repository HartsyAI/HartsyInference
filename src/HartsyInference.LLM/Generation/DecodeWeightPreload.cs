using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Generation;

/// <summary>Uploads a model's decode weights up to a device-memory budget: everything that fits while <paramref name="headroom"/> bytes stay free,
/// in enumeration order, leaving the stragglers to lazy residency. Idempotent, since already-cached tensors are skipped by the backend.</summary>
internal static class DecodeWeightPreload
{
    /// <summary>Preloads <paramref name="weights"/> on <paramref name="backend"/> within its free memory minus <paramref name="headroom"/>; a backend with no device memory does nothing.</summary>
    public static void Preload(IBackend backend, IEnumerable<Tensor> weights, long headroom)
    {
        long free = backend.FreeMemoryBytes();
        if (free <= headroom) return;
        List<Tensor> toPreload = [];
        long budget = free - headroom;
        long skipped = 0;
        int skippedCount = 0;
        foreach (Tensor t in weights)
        {
            long bytes = Tensor.ComputeByteSize(t.Shape, t.DType);
            if (budget - bytes < 0) { skipped += bytes; skippedCount++; continue; }
            budget -= bytes;
            toPreload.Add(t);
        }
        Logs.Info($"[preload] free={backend.FreeMemoryBytes() >> 20}MB headroom={headroom >> 20}MB " +
            $"kept={toPreload.Count} skipped={skippedCount} ({skipped >> 20}MB left lazy)");
        backend.PreloadWeights(toPreload);
    }
}
