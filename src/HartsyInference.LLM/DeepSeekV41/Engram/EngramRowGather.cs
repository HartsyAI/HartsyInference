namespace HartsyInference.LLM.DeepSeekV41.Engram;

/// <summary>Fetches table rows as bf16 bit patterns, the shape of <c>EngramTableStore.Gather</c>.</summary>
/// <param name="rows">Row ids to fetch.</param>
/// <param name="destBf16">Receives <c>rows.Length x rowWidth</c> values.</param>
public delegate void EngramRowGather(ReadOnlySpan<long> rows, Span<ushort> destBf16);
