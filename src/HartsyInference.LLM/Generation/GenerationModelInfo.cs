namespace HartsyInference.LLM.Generation;

/// <summary>Static shape of a generation model that drivers need without knowing its architecture.</summary>
/// <param name="Name">Display name (architecture or checkpoint id).</param>
/// <param name="VocabSize">Width of the logits rows <see cref="IGenerationModel.ProjectLogits"/> returns.</param>
/// <param name="HiddenSize">Width of the hidden rows <see cref="IGenerationModel.Prefill"/> returns.</param>
/// <param name="NumLayers">Decoder layer count.</param>
/// <param name="MaxContextTokens">Longest sequence the model was trained for.</param>
public sealed record GenerationModelInfo(string Name, int VocabSize, int HiddenSize, int NumLayers, int MaxContextTokens);
