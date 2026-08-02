using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Cuda;

namespace HartsyInference.Engine.Placement;

/// <summary>One contiguous layer range assigned to one device selector; ranges are [Start, End).</summary>
public readonly record struct LlmStagePlan(string Device, int StartLayer, int EndLayer);

/// <summary>Turns a <see cref="PlacementConfig"/> plus probed device topology into concrete assignments: layer
/// ranges for LLM sharding, component devices for diffusion. Explicit user config always wins; auto plans fill
/// the gaps from free VRAM.</summary>
public static class PlacementPlanner
{
    /// <summary>VRAM held back per device when auto-splitting by free memory: activations, KV, workspaces.</summary>
    private const long PerDeviceReserveBytes = 2L << 30;

    /// <summary>Splits <paramref name="layerCount"/> layers across <paramref name="shardDevices"/> proportionally
    /// to <paramref name="ratios"/> (explicit, llama.cpp tensor-split style) or to probed free VRAM minus a fixed
    /// reserve. The last stage is additionally charged <paramref name="lastStageExtraBytes"/> (final norm + lm_head
    /// + logits) when planning by VRAM. Every stage gets at least one layer.</summary>
    public static IReadOnlyList<LlmStagePlan> LlmSplitPlan(
        IReadOnlyList<string> shardDevices,
        IReadOnlyList<float>? ratios,
        int layerCount,
        long perLayerBytes,
        long lastStageExtraBytes = 0)
    {
        ArgumentNullException.ThrowIfNull(shardDevices);
        if (shardDevices.Count == 0)
        {
            throw new ArgumentException("At least one shard device is required.", nameof(shardDevices));
        }
        if (layerCount < shardDevices.Count)
        {
            throw new ArgumentException(
                $"Cannot split {layerCount} layers across {shardDevices.Count} devices — fewer layers than stages.",
                nameof(layerCount));
        }
        if (ratios is not null && ratios.Count != shardDevices.Count)
        {
            throw new ArgumentException(
                $"ShardRatios has {ratios.Count} entries but ShardDevices has {shardDevices.Count}.", nameof(ratios));
        }

        float[] weights = new float[shardDevices.Count];
        if (ratios is not null)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = ratios[i] > 0 ? ratios[i] : 0f;
            }
        }
        else
        {
            IReadOnlyList<GpuTopologyInfo> topology = CudaTopology.Probe();
            for (int i = 0; i < shardDevices.Count; i++)
            {
                int ordinal = BackendFactory.ParseOrdinal(shardDevices[i]);
                long free = 0;
                foreach (GpuTopologyInfo gpu in topology)
                {
                    if (gpu.Ordinal == ordinal)
                    {
                        free = Math.Max(0, gpu.FreeMemoryBytes - PerDeviceReserveBytes);
                        break;
                    }
                }
                // Charge the head/logits cost to the last stage by shrinking its budget before ratioing.
                if (i == shardDevices.Count - 1)
                {
                    free = Math.Max(0, free - lastStageExtraBytes);
                }
                weights[i] = free;
            }
        }

        float total = 0;
        foreach (float w in weights)
        {
            total += w;
        }
        if (total <= 0)
        {
            Logs.Warning("[Placement] No usable VRAM signal for the layer split — falling back to an even split.");
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = 1;
            }
            total = weights.Length;
        }

        // Largest-remainder assignment with a 1-layer floor per stage.
        int[] counts = new int[weights.Length];
        int assigned = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            counts[i] = Math.Max(1, (int)MathF.Floor(layerCount * (weights[i] / total)));
            assigned += counts[i];
        }
        // Distribute the remainder (or claw back overshoot) at the highest-weight stages first.
        int[] order = [.. Enumerable.Range(0, weights.Length).OrderByDescending(i => weights[i])];
        int guard = 0;
        while (assigned != layerCount && guard++ < 4 * layerCount)
        {
            foreach (int i in order)
            {
                if (assigned < layerCount)
                {
                    counts[i]++;
                    assigned++;
                }
                else if (assigned > layerCount && counts[i] > 1)
                {
                    counts[i]--;
                    assigned--;
                }
                if (assigned == layerCount)
                {
                    break;
                }
            }
        }

        List<LlmStagePlan> plan = new(weights.Length);
        int start = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            plan.Add(new LlmStagePlan(shardDevices[i], start, start + counts[i]));
            start += counts[i];
        }
        Logs.Info($"[Placement] LLM layer split ({layerCount} layers, ~{perLayerBytes >> 20} MB/layer): "
            + string.Join(", ", plan.Select(p => $"{p.Device}=[{p.StartLayer},{p.EndLayer})")));
        return plan;
    }

    /// <summary>Auto component placement for diffusion on a multi-GPU box: text encoders on the smallest-VRAM
    /// device, denoiser + VAE on the largest. Single (or no) GPU → <see cref="PlacementConfig.Single"/>.
    /// Merged UNDER any explicit user config by the caller — this only produces the auto suggestion.</summary>
    public static PlacementConfig DiffusionAutoPlan()
    {
        IReadOnlyList<GpuTopologyInfo> topology = CudaTopology.Probe();
        if (topology.Count < 2)
        {
            return PlacementConfig.Single;
        }
        GpuTopologyInfo smallest = topology[0];
        GpuTopologyInfo largest = topology[0];
        foreach (GpuTopologyInfo gpu in topology)
        {
            if (gpu.TotalMemoryBytes < smallest.TotalMemoryBytes)
            {
                smallest = gpu;
            }
            if (gpu.TotalMemoryBytes > largest.TotalMemoryBytes)
            {
                largest = gpu;
            }
        }
        if (smallest.Ordinal == largest.Ordinal)
        {
            return PlacementConfig.Single;
        }
        return new PlacementConfig { TextEncoderDevice = $"cuda:{smallest.Ordinal}" };
    }
}
