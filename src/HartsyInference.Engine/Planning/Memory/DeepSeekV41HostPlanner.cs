using HartsyInference.Core.MemoryManagement;
using HartsyInference.Engine.Placement;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.DeepSeekV41;

namespace HartsyInference.Engine.Planning.Memory;

/// <summary>The host residency plan of a DeepSeek-V4.1 checkpoint, from its headers alone: the working set the Text path charges to RAM and the stored weights it maps.</summary>
internal static class DeepSeekV41HostPlanner
{
    /// <summary>Free RAM a load keeps beyond its working set, so the rest of the process is not starved.</summary>
    internal const long HostHeadroomBytes = 4L * 1024 * 1024 * 1024;

    /// <summary>Plans the checkpoint at <paramref name="directory"/> against <paramref name="availableBytes"/> of free host RAM. Opens headers only.</summary>
    internal static ResidencyPlan Plan(string directory, long availableBytes)
    {
        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(directory);
        return ResidencyPlanner.Plan(Demand(checkpoint), availableBytes);
    }

    /// <summary>The demand of a checkpoint under the options the Text path loads with.</summary>
    internal static ResidencyDemand Demand(DeepSeekV41Checkpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        DeepSeekV41Config cfg = checkpoint.Config;
        DeepSeekV41LoadOptions options = HfTextDirectoryLoader.LoadOptions;
        IReadOnlyDictionary<DeepSeekV41WeightClass, long> bytes = checkpoint.Weights.BytesByClass;
        long dense = bytes[DeepSeekV41WeightClass.Dense] + bytes[DeepSeekV41WeightClass.Embed] + bytes[DeepSeekV41WeightClass.Head];
        long widened = options.Residency == DeepSeekV41Residency.WidenedF32 ? 4 * dense : 0;
        return new ResidencyDemand
        {
            DenseBytes = dense,
            ExpertBytes = bytes[DeepSeekV41WeightClass.Expert],
            EngramBytes = bytes[DeepSeekV41WeightClass.Engram],
            WorkingSet = new Dictionary<ResidencyAccount, long>
            {
                [ResidencyAccount.DenseWeights] = widened + DeepSeekV41WorkingMemory.SmallTensorBytes(cfg),
                [ResidencyAccount.RoutedExperts] = DeepSeekV41WorkingMemory.ExpertCacheBytes(cfg, options),
                [ResidencyAccount.Engram] = DeepSeekV41WorkingMemory.EngramCacheBytes(cfg, options),
                [ResidencyAccount.ConversionWorkspace] = DeepSeekV41WorkingMemory.RowWindowBytes,
                [ResidencyAccount.KvCache] = DeepSeekV41WorkingMemory.SequenceStateBytes(cfg, options.MaxTokens),
                [ResidencyAccount.Activations] = DeepSeekV41WorkingMemory.ActivationBytes(cfg, options.MaxTokens),
            },
            HeadroomBytes = HostHeadroomBytes,
        };
    }
}
