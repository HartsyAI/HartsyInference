using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Cuda;

namespace HartsyInference.Engine.Placement;

/// <summary>One contiguous layer range assigned to one device selector; ranges are [Start, End).</summary>
public readonly record struct LlmStagePlan(string Device, int StartLayer, int EndLayer);

/// <summary>Turns a <see cref="PlacementConfig"/> plus probed device topology into concrete assignments: layer ranges for LLM sharding, component devices for diffusion. Explicit user config always wins; auto plans fill the gaps from free VRAM.</summary>
public static class PlacementPlanner
{
    /// <summary>VRAM held back per device when auto-splitting by free memory: activations, KV, workspaces.</summary>
    private const long PerDeviceReserveBytes = 2L << 30;

    /// <summary>Splits <paramref name="layerCount"/> layers across <paramref name="shardDevices"/> proportionally to <paramref name="ratios"/> (explicit, llama.cpp tensor-split style) or to probed free VRAM minus a fixed reserve. The last stage is additionally charged <paramref name="lastStageExtraBytes"/> (final norm + lm_head + logits) when planning by VRAM. Every stage gets at least one layer.</summary>
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

    /// <summary>Auto component placement for diffusion on a multi-GPU box: text encoders on the smallest-VRAM device, denoiser + VAE on the largest. Single (or no) GPU → <see cref="PlacementConfig.Single"/>. Merged UNDER any explicit user config by the caller — this only produces the auto suggestion.</summary>
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

    /// <summary>Splits a DiT's block loop into an N-way range split (Phase 8, generalized from the original 2-way shape) proportional to each stage backend's free VRAM minus the fixed reserve. <paramref name="sharedWeightBytesFirstStage"/> (img_in/time-embed/text-fusion/final-layer — <c>EnumerateSharedWeights</c>) is charged to the FIRST stage's budget since those weights always live there — the mirror of <see cref="LlmSplitPlan"/>'s <c>lastStageExtraBytes</c>, but as a prefix cost instead of a suffix one. Returns per-stage block counts (parallel to <paramref name="freeBytesPerStage"/>) via <see cref="LargestRemainderCounts"/>, which already generalizes to N weights — callers turn counts into <c>[start, end)</c> ranges via a running prefix sum. The N=2 case (<c>counts[0]</c> as a single split point) is exactly the original 2-way behavior.</summary>
    public static IReadOnlyList<int> DitSplitPlan(IReadOnlyList<long> freeBytesPerStage, int blockCount, long sharedWeightBytesFirstStage)
    {
        ArgumentNullException.ThrowIfNull(freeBytesPerStage);
        if (freeBytesPerStage.Count < 2)
        {
            throw new ArgumentException("DiT sharding needs at least 2 stages.", nameof(freeBytesPerStage));
        }
        if (blockCount < freeBytesPerStage.Count)
        {
            throw new ArgumentException(
                $"Cannot split {blockCount} blocks across {freeBytesPerStage.Count} stages — fewer blocks than stages.",
                nameof(blockCount));
        }

        float[] weights = new float[freeBytesPerStage.Count];
        for (int i = 0; i < weights.Length; i++)
        {
            float w = freeBytesPerStage[i] - PerDeviceReserveBytes;
            if (i == 0)
            {
                w -= sharedWeightBytesFirstStage;
            }
            weights[i] = Math.Max(0, w);
        }
        float total = 0;
        foreach (float w in weights)
        {
            total += w;
        }
        if (total <= 0)
        {
            Logs.Warning("[Placement] No usable VRAM signal for the DiT block split — falling back to an even split.");
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = 1;
            }
            total = weights.Length;
        }

        int[] counts = LargestRemainderCounts(weights, total, blockCount);
        Logs.Info($"[Placement] DiT block split ({blockCount} blocks, {weights.Length} stages): " + DescribeRanges(counts));
        return counts;
    }

    /// <summary>Byte-weighted variant of <see cref="DitSplitPlan(IReadOnlyList{long},int,long)"/> for heterogeneous-block DiTs (Chroma/HunyuanImage/Flux double blocks are ~2× their single blocks — a count-proportional split misallocates by GBs there). Picks each of the <c>stages-1</c> interior split points independently by walking a prefix sum over <paramref name="perBlockBytes"/> to the block index closest to that boundary's cumulative VRAM-share target, then converts to per-stage counts. For N=2 this is exactly the original single-cutpoint prefix-sum search (same tie-break: first-found minimum, ascending split order), so a 2-stage call reproduces the original split point bit-for-bit. Every stage gets at least one block.</summary>
    public static IReadOnlyList<int> DitSplitPlan(IReadOnlyList<long> freeBytesPerStage, IReadOnlyList<long> perBlockBytes, long sharedWeightBytesFirstStage)
    {
        ArgumentNullException.ThrowIfNull(freeBytesPerStage);
        ArgumentNullException.ThrowIfNull(perBlockBytes);
        int stageCount = freeBytesPerStage.Count;
        if (stageCount < 2)
        {
            throw new ArgumentException("DiT sharding needs at least 2 stages.", nameof(freeBytesPerStage));
        }
        int blockCount = perBlockBytes.Count;
        if (blockCount < stageCount)
        {
            throw new ArgumentException(
                $"Cannot split {blockCount} blocks across {stageCount} stages — fewer blocks than stages.",
                nameof(perBlockBytes));
        }

        double[] budget = new double[stageCount];
        double totalBudget = 0;
        for (int i = 0; i < stageCount; i++)
        {
            double b = freeBytesPerStage[i] - PerDeviceReserveBytes;
            if (i == 0)
            {
                b -= sharedWeightBytesFirstStage;
            }
            budget[i] = Math.Max(0, b);
            totalBudget += budget[i];
        }
        long totalBytes = 0;
        foreach (long bytes in perBlockBytes)
        {
            totalBytes += bytes;
        }
        if (totalBudget <= 0)
        {
            Logs.Warning("[Placement] No usable VRAM signal for the DiT block split — falling back to an even byte split.");
            for (int i = 0; i < budget.Length; i++)
            {
                budget[i] = 1;
            }
            totalBudget = stageCount;
        }

        long[] prefixSums = new long[blockCount + 1];
        for (int i = 0; i < blockCount; i++)
        {
            prefixSums[i + 1] = prefixSums[i] + perBlockBytes[i];
        }

        // Each interior boundary's target is the CUMULATIVE VRAM share up to that stage — searched independently
        // over the full prefix-sum array, then clamped to keep boundaries ascending with a 1-block floor per stage
        // (both sides). For stageCount=2 there is exactly one boundary and this degenerates to the original
        // single-cutpoint search over the same [1, blockCount-1] range with the same strict-less-than tie-break.
        int[] splits = new int[stageCount - 1];
        double cumBudget = 0;
        int prevSplit = 0;
        for (int s = 0; s < stageCount - 1; s++)
        {
            cumBudget += budget[s];
            double targetCum = totalBytes * (cumBudget / totalBudget);
            int minSplit = prevSplit + 1;
            int maxSplit = blockCount - (stageCount - 1 - s);
            int bestSplit = minSplit;
            double bestDistance = double.MaxValue;
            for (int split = minSplit; split <= maxSplit; split++)
            {
                double distance = Math.Abs(prefixSums[split] - targetCum);
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    bestSplit = split;
                }
            }
            splits[s] = bestSplit;
            prevSplit = bestSplit;
        }

        int[] counts = new int[stageCount];
        int start = 0;
        for (int s = 0; s < stageCount - 1; s++)
        {
            counts[s] = splits[s] - start;
            start = splits[s];
        }
        counts[stageCount - 1] = blockCount - start;

        Logs.Info($"[Placement] DiT byte-weighted block split ({blockCount} blocks, {totalBytes >> 20} MB, {stageCount} stages): "
            + DescribeRanges(counts));
        return counts;
    }

    /// <summary>Renders per-stage block counts as <c>[start,end)</c> ranges for log lines.</summary>
    private static string DescribeRanges(IReadOnlyList<int> counts)
    {
        List<string> parts = new(counts.Count);
        int start = 0;
        for (int i = 0; i < counts.Count; i++)
        {
            parts.Add($"stage{i}=[{start},{start + counts[i]})");
            start += counts[i];
        }
        return string.Join(", ", parts);
    }

    /// <summary>Rejects <see cref="PlacementConfig"/> combinations that were never designed to compose. Called at every point a placement takes effect (construction and <c>InferenceEngine.SetPlacement</c>) so an invalid config fails fast instead of misbehaving silently at generation time.</summary>
    public static void ValidatePlacement(PlacementConfig placement)
    {
        if (placement.TensorParallelDegree < 1)
        {
            throw new ArgumentException(
                $"TensorParallelDegree must be >= 1 (1 = off), got {placement.TensorParallelDegree}.", nameof(placement));
        }
        if (placement.TensorParallelDegree > 1)
        {
            if (placement.ShardDevices.Count != placement.TensorParallelDegree)
            {
                throw new ArgumentException(
                    $"TensorParallelDegree={placement.TensorParallelDegree} requires exactly that many ShardDevices "
                    + $"(the TP rank devices), got {placement.ShardDevices.Count}. Under tensor parallelism "
                    + "ShardDevices lists the ranks and the layer split is off.", nameof(placement));
            }
            if (placement.ShardRatios is not null)
            {
                throw new ArgumentException(
                    "ShardRatios cannot be set with TensorParallelDegree > 1 — ratios steer the LAYER split, which "
                    + "is off under tensor parallelism (every rank holds the full layer stack; there is nothing to "
                    + "ratio). Leave it null.", nameof(placement));
            }
            if (placement.EnableDitSharding)
            {
                throw new ArgumentException(
                    "TensorParallelDegree > 1 and EnableDitSharding cannot both be set — TP claims ShardDevices as "
                    + "its rank list, while DiT sharding claims it as a block-range pipeline; they were not designed "
                    + "to compose. Configure only one.", nameof(placement));
            }
            if (placement.CfgParallelDevice is not null)
            {
                throw new ArgumentException(
                    "TensorParallelDegree > 1 and CfgParallelDevice cannot both be set — they are two different ways "
                    + "to spend extra GPUs on one model (intra-layer weight split vs concurrent CFG branch) and were "
                    + "not designed to compose. Configure only one.", nameof(placement));
            }
            if (placement.ContextParallelDevices.Count > 0)
            {
                throw new ArgumentException(
                    "TensorParallelDegree > 1 and ContextParallelDevices cannot both be set — TP splits weights "
                    + "across ranks, context parallelism splits the token sequence with replicated weights; they "
                    + "were not designed to compose. Configure only one.", nameof(placement));
            }
        }
        if (placement.ContextParallelDevices.Count == 1)
        {
            throw new ArgumentException(
                "ContextParallelDevices needs at least 2 entries when set — context parallelism splits the token "
                + "sequence across ranks, and a single rank is just the primary backend. Leave it empty to disable.",
                nameof(placement));
        }
        if (placement.ContextParallelDevices.Count >= 2)
        {
            if (placement.EnableDitSharding)
            {
                throw new ArgumentException(
                    "ContextParallelDevices and EnableDitSharding cannot both be set — context parallelism replicates "
                    + "the DiT's weights on every rank for latency, while DiT sharding splits the block range for "
                    + "pooled VRAM; they were not designed to compose. Configure only one.", nameof(placement));
            }
            if (placement.CfgParallelDevice is not null)
            {
                throw new ArgumentException(
                    "ContextParallelDevices and CfgParallelDevice cannot both be set — they are two different ways "
                    + "to spend a second GPU on the same denoiser forward (sequence split vs concurrent CFG branch) "
                    + "and were not designed to compose. Configure only one.", nameof(placement));
            }
        }
        if (!placement.EnableDitSharding)
        {
            return;
        }
        if (placement.ShardDevices.Count < 2)
        {
            throw new ArgumentException(
                $"EnableDitSharding requires at least 2 ShardDevices (the transformer's ForwardSharded block-range "
                + $"split needs at least a 2-stage pipeline), got {placement.ShardDevices.Count}. N>2 is currently "
                + $"only consumed by the Qwen-Image recipe — the other sharded families still read only "
                + $"ShardDevices[0]/[1] (ROADMAP.md item 7).", nameof(placement));
        }
        if (placement.CfgParallelDevice is not null)
        {
            throw new ArgumentException(
                "EnableDitSharding and CfgParallelDevice cannot both be set — they are two different ways to use a "
                + "second GPU for the same model (VRAM pooling vs weight replication for latency) and were not "
                + "designed to compose. Configure only one.", nameof(placement));
        }
    }

    /// <summary>Largest-remainder assignment of <paramref name="itemCount"/> range-partitionable units (LLM layers, DiT blocks) proportional to <paramref name="weights"/>, with a 1-item floor per stage.</summary>
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
