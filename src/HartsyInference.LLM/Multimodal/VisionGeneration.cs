namespace HartsyInference.LLM.Multimodal;

/// <summary>A vision answer: the generated text, and whether decoding ended on a stop token (false: the token budget ran out first).</summary>
public readonly record struct VisionGeneration(string Text, bool StoppedOnToken);
