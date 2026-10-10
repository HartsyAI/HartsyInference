using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Runtime;
using HartsyInference.Cuda;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Engine.Placement;

/// <summary>Plans a GGUF text model's placement against the CUDA devices as they are now. The load path and the memory-fit check
/// both come here, so the fit a host is told and the placement the load uses cannot disagree.</summary>
public static class TextPlacementProbe
{
    /// <summary>Plans <paramref name="path"/> with <paramref name="deviceKey"/> as the primary device; the other CUDA devices follow,
    /// most free memory first.</summary>
    /// <param name="path">The GGUF checkpoint; only its header is read.</param>
    /// <param name="deviceKey">The primary device, <c>cuda:N</c>.</param>
    /// <param name="mode">The requested placement.</param>
    /// <param name="contextTokens">Tokens the KV cache is sized for.</param>
    /// <param name="includeRedundantSplits">Whether the load keeps split originals beside fused copies.</param>
    /// <exception cref="HartsyInferenceException">The primary device is not a visible CUDA device.</exception>
    public static TextPlacement PlanGguf(string path, string deviceKey, TextPlacementMode mode, int contextTokens,
        bool includeRedundantSplits)
    {
        TextPlacementDemand demand = TextPlacementDemandReader.FromGguf(path, contextTokens, includeRedundantSplits, KvCaches.F16Enabled);
        int primary = BackendFactory.ParseOrdinal(deviceKey);
        if (CudaTopology.ProbeDevice(primary) is not GpuTopologyInfo first)
            throw new HartsyInferenceException($"CUDA device '{deviceKey}' was not found; no placement can be planned.");
        List<TextPlacementDevice> devices = [new(BackendFactory.WithOrdinal("cuda", primary), first.FreeMemoryBytes)];
        long? hostFree = HostMemoryInfo.AvailableBytes();
        // The other devices are probed only when one GPU is not the answer: probing opens a context on each, which a model that
        // fits its own GPU should not pay for.
        TextPlacement single = TextPlacementPlanner.Plan(demand, devices, hostFree, mode);
        if (mode == TextPlacementMode.Gpu || (mode == TextPlacementMode.Auto && single.Feasible && single.Mode == TextPlacementMode.Gpu))
            return single;
        foreach (GpuTopologyInfo gpu in CudaTopology.Probe().Where(g => g.Ordinal != primary).OrderByDescending(static g => g.FreeMemoryBytes))
            devices.Add(new TextPlacementDevice(BackendFactory.WithOrdinal("cuda", gpu.Ordinal), gpu.FreeMemoryBytes));
        return TextPlacementPlanner.Plan(demand, devices, hostFree, mode);
    }
}
