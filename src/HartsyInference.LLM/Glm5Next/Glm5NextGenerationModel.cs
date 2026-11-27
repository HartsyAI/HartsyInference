using HartsyInference.Core.Backends;
using HartsyInference.LLM.Generation;

namespace HartsyInference.LLM.Glm5Next;

/// <summary>Adapts a loaded <see cref="Glm5NextModel"/> to <see cref="IGenerationModel"/> so the shared pipeline can drive it.</summary>
/// <remarks>Owns the model. Text only, one token per decode step, no speculation (the MTP layer is not run). A prefill must start at the sequence's
/// committed length. Not thread-safe: run one sequence at a time.</remarks>
public sealed class Glm5NextGenerationModel : HostReplayGenerationModel
{
    private readonly Glm5NextModel _model;

    /// <param name="model">The loaded model; this adapter takes ownership.</param>
    /// <param name="outputBackend">Backend the model runs on. The hidden and logits tensors it returns are host tensors whatever the backend.</param>
    /// <param name="maxTokens">The longest sequence the model was loaded for.</param>
    public Glm5NextGenerationModel(Glm5NextModel model, IBackend outputBackend, int maxTokens)
        : base("Sage S2.1 Flash", Required(model, nameof(model)).Config.Vocab, model.Config.Hidden, model.Config.Layers, outputBackend, maxTokens,
            "Sage S2.1 Flash", "The Sage S2.1 Flash text path takes token ids only; image rows are not wired yet.", batchDecode: true)
    {
        _model = model;
    }

    /// <summary>An upper bound: per KDA layer the fixed conv history and delta-rule state, per DSA layer a latent row and an indexer key row per context token.</summary>
    public override long EstimateSequenceBytes(int contextTokens)
    {
        Glm5NextConfig c = _model.Config;
        int kda = c.LayerKinds.Count(k => k == Glm5NextAttentionKind.Kda), dsa = c.Layers - kda;
        long kdaFloats = (long)(c.ConvKernel - 1) * 3 * c.KdaKeyDim + (long)c.Heads * c.KdaHeadDim * c.KdaHeadDim;
        long dsaRowFloats = c.KvLoraRank + 2L * c.IndexHeadDim;
        return (kda * kdaFloats + dsa * dsaRowFloats * contextTokens) * sizeof(float);
    }

    /// <inheritdoc />
    protected override HostReplaySequenceState CreateReplayState(int capacity)
    {
        Glm5NextState caches = new(_model.Config, capacity);
        return new HostReplaySequenceState(capacity, _model.Config.Hidden,
            (ids, hidden) => _model.ForwardHidden(_model.Embed(ids), caches).AsSpan().CopyTo(hidden), caches.Reset, caches.Dispose);
    }

    /// <inheritdoc />
    protected override void ProjectRows(ReadOnlySpan<float> hidden, int rows, Span<float> logits) => _model.Project(hidden).AsSpan().CopyTo(logits);

    /// <inheritdoc />
    protected override void DisposeModel() => _model.Dispose();
}
