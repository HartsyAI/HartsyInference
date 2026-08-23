using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Engine.Placement;

/// <summary>The 2-backend DiT-shard split every image/video recipe computes at construction time, once its transformer is loaded and live free VRAM is readable. Callers keep their own <c>Logs.Info</c> — the split point is shared, the wording (and each pipeline's sharding caveats) is not.</summary>
internal static class DitShardPlanner
{
    /// <summary>Total on-device byte footprint of a weight enumeration.</summary>
    internal static long SumWeightBytes(IEnumerable<Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        long bytes = 0;
        foreach (Tensor tensor in weights)
        {
            bytes += tensor.DType.ComputeByteCount(tensor.ElementCount);
        }
        return bytes;
    }

    /// <summary>Split point for a heterogeneous-block DiT (Flux/Chroma/HiDream/HunyuanImage double blocks are ~2× their single blocks, so a count-proportional split misallocates by GBs).</summary>
    internal static int SplitBlockByBytes(
        IBackend primary,
        IBackend shard,
        int blockCount,
        Func<int, int, IEnumerable<Tensor>> enumerateBlockRangeWeights,
        IEnumerable<Tensor> sharedWeights)
    {
        ArgumentNullException.ThrowIfNull(enumerateBlockRangeWeights);
        long[] perBlockBytes = new long[blockCount];
        for (int i = 0; i < blockCount; i++)
        {
            perBlockBytes[i] = SumWeightBytes(enumerateBlockRangeWeights(i, i + 1));
        }
        long sharedWeightBytes = SumWeightBytes(sharedWeights);
        (long freeA, long freeB) = FreeVram(primary, shard);
        return PlacementPlanner.DitSplitPlan([freeA, freeB], perBlockBytes, sharedWeightBytes)[0];
    }

    /// <summary>Split point for a homogeneous-block DiT (SD3/Lumina2/Krea2/MiniMax-H3), where block count is already byte-accurate.</summary>
    internal static int SplitBlockByCount(IBackend primary, IBackend shard, int blockCount, IEnumerable<Tensor> sharedWeights)
    {
        long sharedWeightBytes = SumWeightBytes(sharedWeights);
        (long freeA, long freeB) = FreeVram(primary, shard);
        return PlacementPlanner.DitSplitPlan([freeA, freeB], blockCount, sharedWeightBytes)[0];
    }

    private static (long PrimaryFree, long ShardFree) FreeVram(IBackend primary, IBackend shard)
    {
        ArgumentNullException.ThrowIfNull(primary);
        ArgumentNullException.ThrowIfNull(shard);
        (long freeA, _) = primary.GetVramInfo();
        (long freeB, _) = shard.GetVramInfo();
        return (freeA, freeB);
    }
}
