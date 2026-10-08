namespace HartsyInference.Core.Moe;

/// <summary>One layer of a model: dense or sparse, the state it keeps, and whether it is a draft (speculative) head.</summary>
/// <param name="Index">Position in the topology, 0-based and contiguous.</param>
/// <param name="Moe">Sparse feed-forward, or null for a dense feed-forward layer.</param>
/// <param name="StateKind">Sequence state this layer persists between tokens; several kinds may be combined.</param>
/// <param name="IsDraft">A model-native draft/MTP layer that speculation runs, not the target backbone.</param>
public sealed record SparseLayerDescriptor(int Index, MoeLayerDescriptor? Moe, SequenceStateKind StateKind = SequenceStateKind.StandardKv,
        bool IsDraft = false)
{
    /// <summary>True when the feed-forward is sparse.</summary>
    public bool IsSparse => Moe is not null;
}
