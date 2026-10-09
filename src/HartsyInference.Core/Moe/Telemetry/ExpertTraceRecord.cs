namespace HartsyInference.Core.Moe.Telemetry;

/// <summary>One access in a routing trace: which expert of which layer was routed, and how many bytes it moves when missed.</summary>
public readonly record struct ExpertTraceRecord(ushort Layer, ushort Expert, uint Bytes);
