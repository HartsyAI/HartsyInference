using HartsyInference.Core.Backends;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>One contiguous DiT block range bound to one backend; ranges are <c>[StartBlock, EndBlock)</c>. Mirrors
/// <c>LlmStage</c>'s shape for the diffusion side of Phase 8+ N-way sharding — DiT stages need no KV-cache-per-layer
/// bookkeeping, so this stays a plain triple rather than sharing the LLM type. Stage 0's backend also owns the DiT's
/// shared (non-block) weights and the output head; see <c>QwenImageTransformer.ForwardSharded</c>.</summary>
public sealed record DitShardStage(IBackend Backend, int StartBlock, int EndBlock);
