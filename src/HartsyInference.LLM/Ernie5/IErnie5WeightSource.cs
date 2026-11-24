using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Ernie5;

/// <summary>Where <see cref="Ernie5Model"/> reads its weights: a checkpoint on disk (<see cref="Ernie5CheckpointWeights"/>) or any other
/// store with the checkpoint's key names.</summary>
public interface IErnie5WeightSource : IModelWeightSource;
