namespace HartsyInference.Core.Memory;

/// <summary>Access-pattern hint for <see cref="MmapHandle.Advise"/>; values are the Linux madvise constants.</summary>
public enum MmapAdvice
{
    /// <summary>Default kernel readahead.</summary>
    Normal = 0,

    /// <summary>Random access: no readahead, so a sparse touch of a huge shard does not drag neighbours into the page cache.</summary>
    Random = 1,

    /// <summary>Sequential access: aggressive readahead.</summary>
    Sequential = 2,

    /// <summary>The range will be read soon.</summary>
    WillNeed = 3,

    /// <summary>The range will not be read again, so its clean pages can be dropped.</summary>
    DontNeed = 4,
}
