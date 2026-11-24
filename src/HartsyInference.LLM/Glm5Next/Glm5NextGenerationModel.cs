using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Glm5Next;

/// <summary>Adapts a loaded <see cref="Glm5NextModel"/> to <see cref="IGenerationModel"/> so the shared pipeline can drive it.</summary>
/// <remarks>Owns the model. Text only, one token per decode step, no speculation (the MTP layer is not run). A prefill must start at the sequence's
/// committed length. Not thread-safe: run one sequence at a time.</remarks>
public sealed class Glm5NextGenerationModel : IGenerationModel
{
    private readonly Glm5NextModel _model;

    /// <param name="model">The loaded model; this adapter takes ownership.</param>
    /// <param name="outputBackend">Backend the model runs on. The hidden and logits tensors it returns are host tensors whatever the backend.</param>
    /// <param name="maxTokens">The longest sequence the model was loaded for.</param>
    public Glm5NextGenerationModel(Glm5NextModel model, IBackend outputBackend, int maxTokens)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(outputBackend);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxTokens, 1);
        _model = model;
        OutputBackend = outputBackend;
        MaxTokens = maxTokens;
        Info = new GenerationModelInfo("GLM-5.3-Flash", model.Config.Vocab, model.Config.Hidden, model.Config.Layers, maxTokens);
    }

    /// <summary>The longest sequence the model was loaded for.</summary>
    public int MaxTokens { get; }

    /// <inheritdoc />
    public GenerationModelInfo Info { get; }

    /// <inheritdoc />
    public GenerationCapabilities Capabilities { get; } = new() { SupportsBatchDecode = true };

    /// <inheritdoc />
    public IBackend OutputBackend { get; }

    /// <inheritdoc />
    public ISequenceState CreateSequenceState(SequenceStateOptions options)
    {
        if (options.MaxSequenceTokens > MaxTokens)
            throw new ArgumentOutOfRangeException(nameof(options), options.MaxSequenceTokens, $"The model was loaded for sequences of up to {MaxTokens} tokens.");
        Glm5NextState caches = new(_model.Config, options.MaxSequenceTokens);
        return new HostReplaySequenceState(options.MaxSequenceTokens, _model.Config.Hidden,
            (ids, hidden) => _model.ForwardHidden(_model.Embed(ids), caches).AsSpan().CopyTo(hidden), caches.Reset, caches.Dispose);
    }

    /// <inheritdoc />
    public Tensor Prefill(in PrefillChunk chunk, ISequenceState state) => Prefill(chunk, state, CancellationToken.None);

    /// <summary>Checks <paramref name="cancel"/> before running; a prefill is one forward pass and is not interruptible once started.</summary>
    public Tensor Prefill(in PrefillChunk chunk, ISequenceState state, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        HostReplaySequenceState sequence = AsSequence(state);
        if (chunk.Embeds is not null) throw new NotSupportedException("The GLM-5.3-Flash text path takes token ids only; image rows are not wired yet.");
        ReadOnlySpan<int> ids = chunk.TokenIds.Span;
        if (ids.IsEmpty) throw new ArgumentException("Prefill needs at least one token.", nameof(chunk));
        if (chunk.PosStart != sequence.Length)
            throw new ArgumentException($"The chunk starts at position {chunk.PosStart} but the sequence holds {sequence.Length} tokens.", nameof(chunk));
        foreach (int id in ids)
            if ((uint)id >= (uint)_model.Config.Vocab) throw new ArgumentOutOfRangeException(nameof(chunk), id, "Token id is outside the vocabulary.");

        int dim = _model.Config.Hidden;
        float[] hidden = new float[ids.Length * dim];
        sequence.Append(ids, hidden);
        int rows = chunk.LastRowOnly ? 1 : ids.Length;
        return Hidden(hidden.AsSpan((ids.Length - rows) * dim, rows * dim), rows);
    }

    /// <inheritdoc />
    public Tensor DecodeBatch(ReadOnlySpan<int> tokenIds, ISequenceState[] states)
    {
        ArgumentNullException.ThrowIfNull(states);
        if (states.Length != tokenIds.Length) throw new ArgumentException($"{tokenIds.Length} tokens for {states.Length} sequence states.");
        HashSet<ISequenceState> seen = new(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < tokenIds.Length; i++)
        {
            HostReplaySequenceState sequence = AsSequence(states[i]);
            if (!seen.Add(sequence)) throw new ArgumentException("A sequence state appears twice in one batch.", nameof(states));
            if (sequence.Length >= sequence.Capacity) throw new InvalidOperationException($"Sequence {i} is full ({sequence.Capacity} tokens).");
            if ((uint)tokenIds[i] >= (uint)_model.Config.Vocab) throw new ArgumentOutOfRangeException(nameof(tokenIds), tokenIds[i], "Token id is outside the vocabulary.");
        }
        int dim = _model.Config.Hidden;
        float[] hidden = new float[tokenIds.Length * dim];
        for (int i = 0; i < tokenIds.Length; i++) AsSequence(states[i]).Append(tokenIds.Slice(i, 1), hidden.AsSpan(i * dim, dim));
        return Hidden(hidden, tokenIds.Length);
    }

    /// <inheritdoc />
    public Tensor ProjectLogits(Tensor hidden, int rows)
    {
        ArgumentNullException.ThrowIfNull(hidden);
        int dim = _model.Config.Hidden, vocab = _model.Config.Vocab;
        if (hidden.DType != DType.F32 || hidden.ElementCount < (long)rows * dim)
            throw new ArgumentException($"hidden must be F32 with at least {rows} x {dim} values.", nameof(hidden));
        float[] logits = _model.Project(hidden.AsReadOnlySpan<float>().Slice(0, rows * dim));
        Tensor t = new(new TensorShape(1, rows, vocab), DType.F32);
        logits.AsSpan().CopyTo(t.AsSpan<float>());
        return t;
    }

    /// <summary>Empty: the weights are host arrays and a checkpoint view, not engine tensors, so there is nothing for a residency planner to place.</summary>
    public IEnumerable<Tensor> EnumerateWeights(bool includeRedundantSplits) => [];

    /// <summary>An upper bound: per KDA layer the fixed conv history and delta-rule state, per DSA layer a latent row and an indexer key row per context token.</summary>
    public long EstimateSequenceBytes(int contextTokens)
    {
        Glm5NextConfig c = _model.Config;
        int kda = c.LayerKinds.Count(k => k == Glm5NextAttentionKind.Kda), dsa = c.Layers - kda;
        long kdaFloats = (long)(c.ConvKernel - 1) * 3 * c.KdaKeyDim + (long)c.Heads * c.KdaHeadDim * c.KdaHeadDim;
        long dsaRowFloats = c.KvLoraRank + 2L * c.IndexHeadDim;
        return (kda * kdaFloats + dsa * dsaRowFloats * contextTokens) * sizeof(float);
    }

    /// <summary>Free and total host memory as the runtime sees it, also when the model runs on a GPU: the sequence states live on the host.</summary>
    public CapacitySnapshot Capacity()
    {
        GCMemoryInfo info = GC.GetGCMemoryInfo();
        return new CapacitySnapshot(Math.Max(0, info.TotalAvailableMemoryBytes - info.MemoryLoadBytes), info.TotalAvailableMemoryBytes);
    }

    /// <inheritdoc />
    public void Dispose() => _model.Dispose();

    private Tensor Hidden(ReadOnlySpan<float> values, int rows)
    {
        Tensor t = new(new TensorShape(1, rows, _model.Config.Hidden), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static HostReplaySequenceState AsSequence(ISequenceState state) =>
        state as HostReplaySequenceState ?? throw new ArgumentException($"{state.GetType().Name} is not a GLM-5.3-Flash sequence state.", nameof(state));
}
