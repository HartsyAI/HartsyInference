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

        int[] counts = LargestRemainderCounts(weights, total, layerCount);

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

    /// <summary>Splits a DiT's block loop into a 2-way range split (Phase 8) proportional to each backend's free
    /// VRAM minus the fixed reserve. <paramref name="sharedWeightBytesA"/> (img_in/time-embed/text-fusion/final-layer
    /// — <c>EnumerateSharedWeights</c>) is charged to backend A's budget since those weights always live there —
    /// the mirror of <see cref="LlmSplitPlan"/>'s <c>lastStageExtraBytes</c>, but on the FIRST stage: DiT sharding's
    /// shared weights are a prefix cost, not a suffix one. Returns the split point: blocks <c>[0, result)</c> run
    /// on A, <c>[result, blockCount)</c> on B. The 1-block floor per stage guarantees the result is in
    /// <c>[1, blockCount-1]</c>, the range <c>Krea2Transformer.ForwardSharded</c> requires.</summary>
    public static int DitSplitPlan(long freeBytesA, long freeBytesB, int blockCount, long sharedWeightBytesA)
    {
        if (blockCount < 2)
        {
            throw new ArgumentException("DiT sharding needs at least 2 blocks to split.", nameof(blockCount));
        }

        float weightA = Math.Max(0, freeBytesA - PerDeviceReserveBytes - sharedWeightBytesA);
        float weightB = Math.Max(0, freeBytesB - PerDeviceReserveBytes);
        float total = weightA + weightB;
        if (total <= 0)
        {
            Logs.Warning("[Placement] No usable VRAM signal for the DiT block split — falling back to an even split.");
            weightA = weightB = 1;
            total = 2;
        }

        int[] counts = LargestRemainderCounts([weightA, weightB], total, blockCount);
        Logs.Info($"[Placement] DiT block split ({blockCount} blocks): A=[0,{counts[0]}), B=[{counts[0]},{blockCount})");
        return counts[0];
    }

    /// <summary>Byte-weighted variant of <see cref="DitSplitPlan(long,long,int,long)"/> for heterogeneous-block DiTs
    /// (Chroma/HunyuanImage/Flux double blocks are ~2× their single blocks — a count-proportional split misallocates
    /// by GBs there). Picks the split point whose stage-A byte load best matches A's share of usable VRAM via a
    /// prefix-sum walk over <paramref name="perBlockBytes"/>. Same contract: blocks <c>[0, result)</c> on A,
    /// <c>[result, count)</c> on B, result guaranteed in <c>[1, count-1]</c>.</summary>
    public static int DitSplitPlan(long freeBytesA, long freeBytesB, IReadOnlyList<long> perBlockBytes, long sharedWeightBytesA)
    {
        ArgumentNullException.ThrowIfNull(perBlockBytes);
        int blockCount = perBlockBytes.Count;
        if (blockCount < 2)
        {
            throw new ArgumentException("DiT sharding needs at least 2 blocks to split.", nameof(perBlockBytes));
        }

        double budgetA = Math.Max(0, freeBytesA - PerDeviceReserveBytes - sharedWeightBytesA);
        double budgetB = Math.Max(0, freeBytesB - PerDeviceReserveBytes);
        double totalBudget = budgetA + budgetB;
        long totalBytes = 0;
        foreach (long bytes in perBlockBytes)
        {
            totalBytes += bytes;
        }
        if (totalBudget <= 0)
        {
            Logs.Warning("[Placement] No usable VRAM signal for the DiT block split — falling back to an even byte split.");
            budgetA = budgetB = 1;
            totalBudget = 2;
        }

        double targetA = totalBytes * (budgetA / totalBudget);
        int bestSplit = 1;
        double bestDistance = double.MaxValue;
        long prefix = 0;
        for (int split = 1; split <= blockCount - 1; split++)
        {
            prefix += perBlockBytes[split - 1];
            double distance = Math.Abs(prefix - targetA);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestSplit = split;
            }
        }

        long bytesA = 0;
        for (int i = 0; i < bestSplit; i++)
        {
            bytesA += perBlockBytes[i];
        }
        Logs.Info($"[Placement] DiT block split ({blockCount} blocks, {totalBytes >> 20} MB): "
            + $"A=[0,{bestSplit}) {bytesA >> 20} MB, B=[{bestSplit},{blockCount}) {(totalBytes - bytesA) >> 20} MB");
        return bestSplit;
    }

    /// <summary>Rejects <see cref="PlacementConfig"/> combinations that were never designed to compose. Called at
    /// every point a placement takes effect (construction and <c>InferenceEngine.SetPlacement</c>) so an invalid
    /// config fails fast instead of misbehaving silently at generation time.</summary>
    public static void ValidatePlacement(PlacementConfig placement)
    {
        if (!placement.EnableDitSharding)
        {
            return;
        }
        if (placement.ShardDevices.Count != 2)
        {
            throw new ArgumentException(
                $"EnableDitSharding requires exactly 2 ShardDevices (the transformer's ForwardSharded block-range "
                + $"split needs a backendA/backendB pair), got {placement.ShardDevices.Count}.", nameof(placement));
        }
        if (placement.CfgParallelDevice is not null)
        {
            throw new ArgumentException(
                "EnableDitSharding and CfgParallelDevice cannot both be set — they are two different ways to use a "
                + "second GPU for the same model (VRAM pooling vs weight replication for latency) and were not "
                + "designed to compose. Configure only one.", nameof(placement));
        }
    }

    /// <summary>Largest-remainder assignment of <paramref name="itemCount"/> range-partitionable units (LLM layers,
    /// DiT blocks) proportional to <paramref name="weights"/>, with a 1-item floor per stage.</summary>
    private static int[] LargestRemainderCounts(float[] weights, float total, int itemCount)
    {
        int[] counts = new int[weights.Length];
        int assigned = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            counts[i] = Math.Max(1, (int)MathF.Floor(itemCount * (weights[i] / total)));
            assigned += counts[i];
        }
        // Distribute the remainder (or claw back overshoot) at the highest-weight stages first.
        int[] order = [.. Enumerable.Range(0, weights.Length).OrderByDescending(i => weights[i])];
        int guard = 0;
        while (assigned != itemCount && guard++ < 4 * itemCount)
        {
            foreach (int i in order)
            {
                if (assigned < itemCount)
                {
                    counts[i]++;
                    assigned++;
                }
                else if (assigned > itemCount && counts[i] > 1)
                {
                    counts[i]--;
                    assigned--;
                }
                if (assigned == itemCount)
                {
                    break;
                }
            }
        }
        return counts;
    }
}
