namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Vision tower dimensions from the config, for the checkpoint's <c>vision.*</c> and <c>aligner.*</c> tensors.</summary>
public sealed record DeepSeekV41VisionConfig(int NumLayers, int HiddenSize, int NumHeads, int IntermediateSize, int PatchSize, int DownsampleRatio);
