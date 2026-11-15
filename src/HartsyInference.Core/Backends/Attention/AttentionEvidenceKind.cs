namespace HartsyInference.Core.Backends.Attention;

/// <summary>An observed execution or preparation action; declarations alone are not execution evidence.</summary>
public enum AttentionEvidenceKind
{
    Execution,
    Repeat,
    Permute,
    Cast,
    MaterializeMask,
    PrefixGather,
    Workspace,
    Cache,
    GraphReplay,
    Synchronization,
}
