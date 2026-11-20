namespace HartsyInference.Core.Backends.Attention;

/// <summary>An observed execution or preparation action; declarations alone are not execution evidence.</summary>
/// <remarks>Kinds count different things and are not summed across each other: <see cref="Cast"/>, <see cref="Repeat"/>,
/// <see cref="Permute"/>, <see cref="ExpandMask"/> and <see cref="StagingCopy"/> are bytes a step produced;
/// <see cref="Workspace"/> is bytes a call allocated, which can be the same buffers.</remarks>
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

    /// <summary>A stored mask or bias expanded to the layout a kernel reads (a key-only row repeated per query, a bias
    /// widened to a cache's capacity); bytes are the expanded copy's.</summary>
    ExpandMask,

    /// <summary>The score matrix a row builds in full before the softmax; bytes are its size. The same buffer is also a
    /// <see cref="Workspace"/> record.</summary>
    MaterializeScores,

    /// <summary>A call a device backend ran on the host reference through a declared row; bytes are the operands and
    /// output it read and wrote on the host.</summary>
    HostReference,

    /// <summary>Operands copied into the contiguous layout an entry needs, or an output copied back out of it, when no
    /// row reads the view in place; bytes are the copy's.</summary>
    StagingCopy,
}
