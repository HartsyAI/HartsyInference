using HartsyInference.Core.Backends;
using HartsyInference.LLM.Generation;

namespace HartsyInference.LLM.Hy4Flash;

/// <summary>Adapts a loaded Hy4 Flash text model to <see cref="IGenerationModel"/> so the shared pipeline can drive it.</summary>
/// <remarks>Owns the model. Text only: a prefill that carries embeddings is refused, because an image placeholder run needs its features, which only the vision path supplies.
/// One token per decode step, no speculation. A prefill must start at the sequence's committed length. Not thread-safe: run one sequence at a time.</remarks>
public sealed class Hy4FlashGenerationModel : HostReplayGenerationModel
{
    private readonly Hy4FlashModel _model;

    /// <param name="model">The loaded model; this adapter takes ownership.</param>
    /// <param name="outputBackend">Backend the model runs on. The hidden and logits tensors it returns are host tensors whatever the backend.</param>
    /// <param name="maxTokens">The longest sequence the model was loaded for.</param>
    public Hy4FlashGenerationModel(Hy4FlashModel model, IBackend outputBackend, int maxTokens)
        : base("Hy4 Flash", Required(model, nameof(model)).Config.VocabSize, model.Config.HiddenSize, model.Config.NumHiddenLayers,
            outputBackend, maxTokens, "Hy4 Flash", "The Hy4 Flash text path takes text tokens only; image input runs through the vision path.", batchDecode: true)
    {
        _model = model;
    }

    /// <summary>The per-position state every layer keeps (latent row and indexer key), for <paramref name="contextTokens"/> positions. The Engram token history (one int per position) is not counted.</summary>
    public override long EstimateSequenceBytes(int contextTokens)
    {
        Hy4FlashConfig c = _model.Config;
        long perPosition = (long)c.KvLoraRank + c.QkRopeHeadDim + c.IndexHeadDim;
        return (long)contextTokens * c.NumHiddenLayers * perPosition * sizeof(float);
    }

    /// <inheritdoc />
    protected override HostReplaySequenceState CreateReplayState(int capacity)
    {
        Hy4FlashSequenceState caches = _model.CreateState();
        return new HostReplaySequenceState(capacity, _model.Config.HiddenSize,
            (ids, hidden) => _model.ForwardHidden(ids, caches, allRows: true).AsSpan().CopyTo(hidden),
            caches.Reset, caches.Dispose);
    }

    /// <inheritdoc />
    protected override void ProjectRows(ReadOnlySpan<float> hidden, int rows, Span<float> logits) => _model.Logits(hidden, rows).AsSpan().CopyTo(logits);

    /// <inheritdoc />
    protected override void DisposeModel() => _model.Dispose();
}
