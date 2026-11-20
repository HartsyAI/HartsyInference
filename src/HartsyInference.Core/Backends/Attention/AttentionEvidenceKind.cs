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

    /// <summary>Scratch a call allocated, with its logical bytes and buffer count. Recorded by host code: a call captured
    /// into a graph records it once, at capture, and its replays record only <see cref="GraphReplay"/> although the graph's
    /// memory nodes allocate again each time. A persistent workspace records its growth, not each use.</summary>
    Workspace,
    Cache,

    /// <summary>One launch of a captured graph; the scratch of the calls inside it is not recorded again.</summary>
    GraphReplay,
    Synchronization,
}
