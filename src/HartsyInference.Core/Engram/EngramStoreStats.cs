namespace HartsyInference.Core.Engram;

/// <summary>Counters of an <see cref="EngramTableStore"/> since it was created.</summary>
/// <param name="PrefetchCalls">Calls to <c>PrefetchAsync</c>.</param>
/// <param name="GatherCalls">Calls to <c>Gather</c>.</param>
/// <param name="RowsRequested">Rows asked for across both, duplicates included.</param>
/// <param name="UniqueRows">Rows left after per-call de-duplication that were checked against the cache.</param>
/// <param name="CacheHits">Unique rows already resident when a load was planned. A Gather whose rows are all resident is served without planning a load and does not count here.</param>
/// <param name="CacheMisses">Unique rows read from the source.</param>
/// <param name="GatherMisses">Rows a <c>Gather</c> had to read itself because nothing prefetched them.</param>
/// <param name="Evictions">Rows dropped to make room.</param>
/// <param name="ReadRanges">Merged reads issued (one per run of rows sharing a 4 KB page).</param>
/// <param name="BytesRead">Bytes read from the source.</param>
/// <param name="PagesTouched">4 KB pages the merged reads span.</param>
/// <param name="ResidentRows">Rows currently cached.</param>
/// <param name="ResidentBytes">Packed bytes those rows occupy.</param>
public readonly record struct EngramStoreStats(
    long PrefetchCalls,
    long GatherCalls,
    long RowsRequested,
    long UniqueRows,
    long CacheHits,
    long CacheMisses,
    long GatherMisses,
    long Evictions,
    long ReadRanges,
    long BytesRead,
    long PagesTouched,
    long ResidentRows,
    long ResidentBytes);
