namespace HartsyInference.Engine.Diagnostics;
/// <summary>Allocation-free timing event. Timestamp uses Stopwatch ticks; Count is prompt/output tokens where applicable.</summary>
public readonly record struct InferenceDiagnosticEvent(long RequestId, InferenceDiagnosticKind Kind, long Timestamp, int Count = 0);
