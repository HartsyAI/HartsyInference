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
        IReadOnlyList<GpuTopologyInfo> gpus = CudaTopology.Probe();
        int primary = BackendFactory.ParseOrdinal(deviceKey);
        List<TextPlacementDevice> devices = [];
        foreach (GpuTopologyInfo gpu in gpus.Where(g => g.Ordinal == primary))
            devices.Add(new TextPlacementDevice(BackendFactory.WithOrdinal("cuda", gpu.Ordinal), gpu.FreeMemoryBytes));
        if (devices.Count == 0)
            throw new HartsyInferenceException($"CUDA device '{deviceKey}' was not found; no placement can be planned.");
        foreach (GpuTopologyInfo gpu in gpus.Where(g => g.Ordinal != primary).OrderByDescending(static g => g.FreeMemoryBytes))
            devices.Add(new TextPlacementDevice(BackendFactory.WithOrdinal("cuda", gpu.Ordinal), gpu.FreeMemoryBytes));
        return TextPlacementPlanner.Plan(demand, devices, HostMemoryInfo.AvailableBytes(), mode);
    }
}
