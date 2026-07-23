using System.Collections.Concurrent;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.LLM.Embeddings;

namespace HartsyInference.Engine.Services;

/// <summary>Decoder-LLM-backed text embeddings (Qwen3-Embedding, gte-Qwen2, e5-mistral, LLM2Vec, …) — reuses the
/// same GGUF-loading + tokenizer pipeline chat models use, via <see cref="DecoderEmbeddingModel"/>. BERT-family
/// encoders (bge/gte/nomic) are NOT wired here: they need a WordPiece tokenizer this service doesn't have
/// (<see cref="HartsyInference.LLM.Embeddings.BertEmbeddingModel"/> exists but isn't reachable from any service
/// yet) — a separate, later scope.</summary>
public sealed class EmbeddingService : IEmbeddingService, IDisposable
{
    private readonly InferenceEngine _engine;

    /// <summary>One loaded model per resolved checkpoint path — a much lighter cache than the audio/diffusion
    /// pipelines' since a GGUF embedding load is a single synchronous file read, not a multi-asset async build.</summary>
    private readonly ConcurrentDictionary<string, DecoderEmbeddingModel> _models = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Creates the service bound to its owning engine.</summary>
    internal EmbeddingService(InferenceEngine engine) => _engine = engine;

    /// <inheritdoc/>
    public Task<EmbeddingResult> GenerateAsync(ModelSpec spec, EmbeddingRequest request, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Input.Count == 0)
            throw new ArgumentException("No input strings supplied to embed.", nameof(request));

        string? path = spec.LocalPath;
        if (string.IsNullOrEmpty(path))
        {
            throw new HartsyInferenceException(
                "Embedding model has no local path. Pass a .gguf file via the model spec (looked under " +
                $"'{RepoPaths.ModelsRoot()}').");
        }

        DecoderEmbeddingModel model = _models.GetOrAdd(path, p => DecoderEmbeddingModel.Load(p));
        IBackend backend = _engine.Backend;

        float[][] vectors = new float[request.Input.Count][];
        int totalTokens = 0;
        for (int i = 0; i < request.Input.Count; i++)
        {
            cancel.ThrowIfCancellationRequested();
            int[] ids = model.Tokenizer.EncodeOrdinary(request.Input[i]);
            // The model pools whichever token is literally LAST in ids -- append EOS (when the tokenizer has
            // one) so that IS the pooled position, matching the reference convention (its tokenizer appends EOS
            // before pooling that same last position). Dropping this silently pools the last real content token
            // instead and diverges from the reference — see the embeddings correctness test for the real check.
            if (model.Tokenizer.EosId is { } eos)
            {
                int[] withEos = new int[ids.Length + 1];
                Array.Copy(ids, withEos, ids.Length);
                withEos[ids.Length] = eos;
                ids = withEos;
            }
            totalTokens += ids.Length;
            vectors[i] = model.Encode(backend, ids);
        }

        return Task.FromResult(new EmbeddingResult { Vectors = vectors, Dimensions = model.Hidden, TotalTokens = totalTokens });
    }

    /// <summary>Releases every cached model — called from <c>InferenceEngine.ReleaseLoaded</c> alongside the
    /// other lazily-constructed services, so a backend switch/free-memory doesn't leave a model bound to a
    /// disposed device.</summary>
    public void Dispose()
    {
        foreach (DecoderEmbeddingModel model in _models.Values)
            model.Dispose();
        _models.Clear();
    }
}
