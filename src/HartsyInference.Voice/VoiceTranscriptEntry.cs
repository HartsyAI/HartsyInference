using HartsyInference.Engine.Requests;

namespace HartsyInference.Voice;

/// <summary>One line of a call's transcript: the caller's words (<see cref="TextRole.User"/>) or the agent's reply
/// (<see cref="TextRole.Assistant"/>), which after a barge-in is the text generated before it.</summary>
public readonly record struct VoiceTranscriptEntry(int TurnId, TextRole Role, string Text, bool Interrupted);
