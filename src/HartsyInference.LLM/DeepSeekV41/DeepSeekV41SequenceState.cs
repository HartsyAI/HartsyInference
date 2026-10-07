using HartsyInference.LLM.DeepSeekV41.Engram;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Everything a V4.1 model keeps for one sequence on the host reference path: each layer's attention cache, the slots shared between layers, and the Engram hash history.</summary>
public sealed class DeepSeekV41SequenceState
{
    /// <summary>One attention cache per layer.</summary>
    public DeepSeekV41AttentionState[] Layers { get; }

    /// <summary>The slots attention layers hand each other; persists across passes like upstream's runtime.</summary>
    public DeepSeekV41SharedAttention Shared { get; } = new();

    /// <summary>Hash history for Engram, or null when the model has no Engram layer.</summary>
    public EngramHasher? Hasher { get; }

    /// <summary>Tokens consumed so far; the next call must start here.</summary>
    public int Length { get; internal set; }

    /// <summary>The most tokens this state can hold.</summary>
    public int Capacity { get; }

    internal DeepSeekV41SequenceState(DeepSeekV41AttentionState[] layers, EngramHasher? hasher, int capacity)
    {
        Layers = layers;
        Hasher = hasher;
        Capacity = capacity;
    }

    /// <summary>Forgets the sequence, ready to start again from position 0.</summary>
    public void Reset()
    {
        foreach (DeepSeekV41AttentionState layer in Layers) layer.Reset();
        Hasher?.Reset();
        Length = 0;
    }
}
