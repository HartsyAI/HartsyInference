using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Generation;

/// <summary>How <see cref="IGenerationModel.CreateSequenceState"/> should size and back a new sequence.</summary>
/// <param name="MaxSequenceTokens">Most tokens the sequence will ever hold (prompt plus generation).</param>
/// <param name="Pool">Shared KV page pool to draw from; null gives the sequence its own fixed buffers.</param>
/// <param name="Loops">Passes of the recurrent block for a looped (latent-recurrent) model such as Dory; 0 takes the model's default and other models ignore it.</param>
public readonly record struct SequenceStateOptions(int MaxSequenceTokens, PagedKvPool? Pool = null, int Loops = 0);
