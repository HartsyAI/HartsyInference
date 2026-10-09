using HartsyInference.Engine.Services;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Engine.Planning.Memory;

/// <summary>Memory estimate for a text checkpoint, read from its safetensors headers without touching a weight.</summary>
/// <remarks>Only DeepSeek-V4.1 directories are described today. The single <see cref="MemoryComponent.LanguageModel"/>
/// phase holds what a decode step reads every token (dense, experts, Engram, embeddings, head); vision and draft
/// weights are reported by class but not counted, because neither is resident unless a caller opts in. Working memory
/// is left at zero: the KV cache and activations depend on the context length the residency planner chooses.</remarks>
internal static class TextMemoryProfile
{
    /// <summary>Whether <paramref name="localPath"/> is a checkpoint this profile can describe.</summary>
    public static bool Handles(string? localPath) =>
        localPath is not null && Directory.Exists(localPath)
        && HfCheckpointDirectory.TryProbe(localPath) is { } info
        && string.Equals(info.ModelType, DeepSeekV41Config.ModelType, StringComparison.Ordinal);

    /// <summary>Reads the headers under <paramref name="directory"/> and sums bytes per weight class.</summary>
    public static MemoryEstimate Estimate(string directory)
    {
        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(directory);
        return Estimate(checkpoint);
    }

    /// <summary>The estimate of an open checkpoint, so a caller that also plans from it opens the headers once.</summary>
    public static MemoryEstimate Estimate(DeepSeekV41Checkpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        IReadOnlyDictionary<DeepSeekV41WeightClass, long> bytes = checkpoint.Weights.BytesByClass;
        long resident = bytes[DeepSeekV41WeightClass.Dense] + bytes[DeepSeekV41WeightClass.Expert]
            + bytes[DeepSeekV41WeightClass.Engram] + bytes[DeepSeekV41WeightClass.Embed]
            + bytes[DeepSeekV41WeightClass.Head];
        long streamFloor = bytes[DeepSeekV41WeightClass.Dense] + bytes[DeepSeekV41WeightClass.Embed]
            + bytes[DeepSeekV41WeightClass.Head];
        // sequence state, prefill activations, small widened tensors and Engram row caches at the sequence length the Text path loads for
        long workingBytes = DeepSeekV41WorkingMemory.AnonymousBytes(checkpoint.Config, 0, HfTextDirectoryLoader.LoadOptions);
        return new MemoryEstimate
        {
            FamilyId = DeepSeekV41Config.ModelType,
            Phases = new[] { new MemoryPhase(MemoryComponent.LanguageModel, resident, workingBytes, streamFloor, Streamable: true) },
            Accuracy = MemoryEstimateAccuracy.HeaderOnly,
            WeightBytesByClass = bytes.ToDictionary(pair => pair.Key.ToString(), pair => pair.Value),
        };
    }
}
