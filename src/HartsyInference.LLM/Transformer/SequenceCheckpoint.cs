namespace HartsyInference.LLM.Transformer;

/// <summary>Cursor-only snapshot of a sequence: the committed length to roll back to, with no tensor data captured.</summary>
public readonly record struct SequenceCheckpoint(int Length);
