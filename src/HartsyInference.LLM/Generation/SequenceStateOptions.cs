using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Generation;

/// <summary>How <see cref="IGenerationModel.CreateSequenceState"/> should size and back a new sequence.</summary>
/// <param name="MaxSequenceTokens">Most tokens the sequence will ever hold (prompt plus generation).</param>
/// <param name="Pool">Shared KV page pool to draw from; null gives the sequence its own fixed buffers.</param>
/// <param name="FullPrecisionKv">Keep KV in F32 even when F16 KV storage is enabled (graph-decode sequences need it).</param>
public readonly record struct SequenceStateOptions(int MaxSequenceTokens, PagedKvPool? Pool = null, bool FullPrecisionKv = false);
