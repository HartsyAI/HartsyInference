using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Generation;

/// <summary>How <see cref="IGenerationModel.CreateSequenceState"/> should size and back a new sequence.</summary>
/// <param name="MaxSequenceTokens">Most tokens the sequence will ever hold (prompt plus generation).</param>
/// <param name="Pool">Shared KV page pool to draw from; null gives the sequence its own fixed buffers.</param>
public readonly record struct SequenceStateOptions(int MaxSequenceTokens, PagedKvPool? Pool = null);
