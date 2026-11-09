using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Ernie5;

/// <summary>Adapts a loaded ERNIE-5.5 host reference model to <see cref="IGenerationModel"/> so the shared pipeline can drive it.</summary>
/// <remarks>Owns the model. Text only, one token per decode step, no speculation. A prefill must start at the sequence's committed length.
/// Not thread-safe: run one sequence at a time.</remarks>
public sealed class Ernie5GenerationModel : IGenerationModel
{
    private readonly Ernie5Model _model;

    /// <param name="model">The loaded model; this adapter takes ownership.</param>
    /// <param name="outputBackend">Backend that hosts the returned hidden and logits tensors (the CPU backend).</param>
    /// <param name="maxTokens">The longest sequence the model was loaded for.</param>
    public Ernie5GenerationModel(Ernie5Model model, IBackend outputBackend, int maxTokens)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(outputBackend);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxTokens, 1);
        _model = model;
        OutputBackend = outputBackend;
        MaxTokens = maxTokens;
        Info = new GenerationModelInfo("ERNIE-5.5", model.Config.VocabSize, model.Config.HiddenSize, model.Config.NumHiddenLayers, maxTokens);
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
        Ernie5SequenceState caches = _model.CreateState(options.MaxSequenceTokens);
        return new HostReplaySequenceState(options.MaxSequenceTokens, _model.Config.HiddenSize,
            (ids, hidden) => _model.ForwardHidden(ids, caches, allRows: true).AsSpan().CopyTo(hidden),
            caches.Reset, caches.Dispose);
    }

    /// <inheritdoc />
    public Tensor Prefill(in PrefillChunk chunk, ISequenceState state) => Prefill(chunk, state, CancellationToken.None);

    /// <summary>Checks <paramref name="cancel"/> before running; a prefill is one forward pass and is not interruptible once started.</summary>
    public Tensor Prefill(in PrefillChunk chunk, ISequenceState state, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        HostReplaySequenceState sequence = AsSequence(state);
        if (chunk.Embeds is not null) throw new NotSupportedException("The ERNIE-5.5 reference model takes text tokens only; image embeddings are not supported.");
        ReadOnlySpan<int> ids = chunk.TokenIds.Span;
        if (ids.IsEmpty) throw new ArgumentException("Prefill needs at least one token.", nameof(chunk));
        if (chunk.PosStart != sequence.Length)
            throw new ArgumentException($"The chunk starts at position {chunk.PosStart} but the sequence holds {sequence.Length} tokens.", nameof(chunk));

        int dim = _model.Config.HiddenSize;
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
        // refuse the whole batch before any sequence moves, as the single-sequence path does
        HashSet<ISequenceState> seen = new(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < tokenIds.Length; i++)
        {
            HostReplaySequenceState sequence = AsSequence(states[i]);
            if (!seen.Add(sequence)) throw new ArgumentException("A sequence state appears twice in one batch.", nameof(states));
            if (sequence.Length >= sequence.Capacity) throw new InvalidOperationException($"Sequence {i} is full ({sequence.Capacity} tokens).");
            if ((uint)tokenIds[i] >= (uint)_model.Config.VocabSize) throw new ArgumentOutOfRangeException(nameof(tokenIds), tokenIds[i], "Token id is outside the vocabulary.");
        }
        int dim = _model.Config.HiddenSize;
        float[] hidden = new float[tokenIds.Length * dim];
        for (int i = 0; i < tokenIds.Length; i++) AsSequence(states[i]).Append(tokenIds.Slice(i, 1), hidden.AsSpan(i * dim, dim));
        return Hidden(hidden, tokenIds.Length);
    }

    /// <inheritdoc />
    public Tensor ProjectLogits(Tensor hidden, int rows)
    {
        ArgumentNullException.ThrowIfNull(hidden);
        int dim = _model.Config.HiddenSize, vocab = _model.Config.VocabSize;
        if (hidden.DType != DType.F32 || hidden.ElementCount < (long)rows * dim)
            throw new ArgumentException($"hidden must be F32 with at least {rows} x {dim} values.", nameof(hidden));
        float[] logits = _model.Logits(hidden.AsReadOnlySpan<float>().Slice(0, rows * dim), rows);
        Tensor t = new(new TensorShape(1, rows, vocab), DType.F32);
        logits.AsSpan().CopyTo(t.AsSpan<float>());
        return t;
    }

    /// <summary>Empty: the weights are host arrays and a checkpoint view, not engine tensors, so there is nothing for a residency planner to place.</summary>
    public IEnumerable<Tensor> EnumerateWeights(bool includeRedundantSplits) => [];

    /// <summary>An upper bound: each layer keeps one row per token of its HCA window and compressed latents, its DSA latent and its indexer keys.</summary>
    public long EstimateSequenceBytes(int contextTokens)
    {
        Ernie5Config c = _model.Config;
        long perTokenFloats = (long)c.HeadDim + c.MlaKvLoraRank + c.MlaQkRopeHeadDim + c.IndexHeadDim;
        return (long)contextTokens * c.NumHiddenLayers * perTokenFloats * sizeof(float);
    }

    /// <summary>Free and total host memory as the runtime sees it; there is no device on this path.</summary>
    public CapacitySnapshot Capacity()
    {
        GCMemoryInfo info = GC.GetGCMemoryInfo();
        return new CapacitySnapshot(Math.Max(0, info.TotalAvailableMemoryBytes - info.MemoryLoadBytes), info.TotalAvailableMemoryBytes);
    }

    /// <inheritdoc />
    public void Dispose() => _model.Dispose();

    private Tensor Hidden(ReadOnlySpan<float> values, int rows)
    {
        Tensor t = new(new TensorShape(1, rows, _model.Config.HiddenSize), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static HostReplaySequenceState AsSequence(ISequenceState state) =>
        state as HostReplaySequenceState ?? throw new ArgumentException($"{state.GetType().Name} is not an ERNIE-5.5 sequence state.", nameof(state));
}
