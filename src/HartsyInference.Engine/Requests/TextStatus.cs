namespace HartsyInference.Engine.Requests;

/// <summary>Progress of a text request before its first token. <paramref name="Phase"/> is loading, queued, prefill or generating; <paramref name="QueuePosition"/> is 0 unless queued.</summary>
public readonly record struct TextStatus(string Phase, int QueuePosition = 0, int PrefillDone = 0, int PrefillTotal = 0);
