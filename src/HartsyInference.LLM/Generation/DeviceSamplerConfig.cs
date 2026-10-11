namespace HartsyInference.LLM.Generation;

/// <summary>Sampling parameters baked into a captured decode graph: top-k candidates (at most the backend's limit), temperature, nucleus and min-p cuts, and the RNG seed.
/// The captured step draws from the device instead of taking an argmax, so a non-greedy request can replay a graph like a greedy one.</summary>
public readonly record struct DeviceSamplerConfig(int TopK, float Temperature, float TopP, float MinP, ulong Seed);
