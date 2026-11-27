using HartsyInference.Core.Backends;
using HartsyInference.LLM.Generation;

namespace HartsyInference.LLM.Maxwell;

/// <summary>Adapts a loaded Maxwell model to <see cref="IGenerationModel"/> so the shared pipeline can drive it.</summary>
/// <remarks>Owns the model. Text only, one token per decode step, no speculation. A prefill must start at the sequence's committed length.
/// Not thread-safe: run one sequence at a time.</remarks>
public sealed class MaxwellGenerationModel : HostReplayGenerationModel
{
    private readonly MaxwellModel _model;

    /// <param name="model">The loaded model; this adapter takes ownership.</param>
    /// <param name="outputBackend">Backend the model runs on. The hidden and logits tensors it returns are host tensors whatever the backend.</param>
    /// <param name="maxTokens">The longest sequence the model was loaded for.</param>
    public MaxwellGenerationModel(MaxwellModel model, IBackend outputBackend, int maxTokens)
        : base("Maxwell-V1.0-Flash", Required(model, nameof(model)).Config.VocabSize, model.Config.HiddenSize, model.Config.NumHiddenLayers,
            outputBackend, maxTokens, "Maxwell", "The Maxwell model takes text tokens only; image embeddings are not supported.", batchDecode: true)
    {
        _model = model;
    }

    /// <summary>The loaded model this adapter drives, for the caller that attaches expert offload to it.</summary>
    public MaxwellModel Model => _model;

    /// <summary>An upper bound on the F32 caches of a sequence of <paramref name="contextTokens"/> tokens: each layer keeps keys, values and indexer
    /// keys, and the caches grow by doubling but never past the state's capacity, which is at most <see cref="HostReplayGenerationModel.MaxTokens"/>.</summary>
    public override long EstimateSequenceBytes(int contextTokens)
    {
        MaxwellConfig c = _model.Config;
        long rowFloats = 2L * c.NumKeyValueHeads * c.HeadDim + c.Sparse.IndexerHeadDim;
        long rows = Math.Max(contextTokens, Math.Min(2L * contextTokens, MaxTokens));
        return rows * c.NumHiddenLayers * rowFloats * sizeof(float);
    }

    /// <inheritdoc />
    protected override HostReplaySequenceState CreateReplayState(int capacity)
    {
        MaxwellSequenceState caches = _model.CreateState(capacity);
        return new HostReplaySequenceState(capacity, _model.Config.HiddenSize,
            (ids, hidden) => _model.ForwardHidden(ids, caches, allRows: true).AsSpan().CopyTo(hidden),
            caches.Reset, caches.Dispose);
    }

    /// <inheritdoc />
    protected override void ProjectRows(ReadOnlySpan<float> hidden, int rows, Span<float> logits) => _model.Logits(hidden, rows).AsSpan().CopyTo(logits);

    /// <inheritdoc />
    protected override void DisposeModel() => _model.Dispose();
}
