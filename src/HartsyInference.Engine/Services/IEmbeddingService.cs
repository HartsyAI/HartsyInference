using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Services;

/// <summary>Typed text-embedding surface: turns input strings into dense, L2-normalized sentence vectors
/// (RAG/semantic-search style). Backed by decoder-LLM GGUF embedding models (Qwen3-Embedding family) today —
/// see <see cref="EmbeddingService"/>'s class doc for the current scope.</summary>
public interface IEmbeddingService
{
    /// <summary>Embeds every string in <paramref name="request"/>.</summary>
    Task<EmbeddingResult> GenerateAsync(ModelSpec spec, EmbeddingRequest request, CancellationToken cancel = default);
}
