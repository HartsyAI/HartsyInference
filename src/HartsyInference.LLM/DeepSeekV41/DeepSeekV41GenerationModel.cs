using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Adapts a loaded V4.1 host reference model to <see cref="IGenerationModel"/> so the shared pipeline can drive it.</summary>
/// <remarks>Built from a <see cref="DeepSeekV41LoadedModel"/>, it owns that model: disposing this releases the checkpoint and row stores. Text only (no image embeds), one token per decode step,
/// no speculation. A prefill must start at the sequence's committed length; the host model runs a later multi-token chunk a token at a time. A refused call changes no sequence. Not thread-safe: run one sequence at a time.</remarks>
public sealed class DeepSeekV41GenerationModel : IGenerationModel
{
    private readonly DeepSeekV41LoadedModel? _loaded;
    private readonly IBackend _backend;
    private readonly DeepSeekV41HostModel _model;
    private readonly int _maxTokens;

    /// <inheritdoc />
    public GenerationModelInfo Info { get; }

    /// <inheritdoc />
    public GenerationCapabilities Capabilities { get; } = new() { SupportsBatchDecode = true };

    /// <inheritdoc />
    public IBackend OutputBackend => _backend;

    /// <param name="loaded">The loaded model; this adapter takes ownership.</param>
    /// <param name="outputBackend">Backend that hosts the returned hidden and logits tensors (the CPU backend).</param>
    public DeepSeekV41GenerationModel(DeepSeekV41LoadedModel loaded, IBackend outputBackend)
    {
        ArgumentNullException.ThrowIfNull(loaded);
        ArgumentNullException.ThrowIfNull(outputBackend);
        _loaded = loaded;
        _backend = outputBackend;
        _model = loaded.Model;
        _maxTokens = loaded.MaxTokens;
        Info = new GenerationModelInfo("DeepSeekV41", _model.VocabSize, _model.Dim, _model.Layers, _maxTokens);
    }

    /// <summary>Adapts a host model whose owner outlives this adapter; disposing this releases nothing.</summary>
    /// <param name="model">The host model. This adapter does not own it.</param>
    /// <param name="maxTokens">The longest sequence the model was loaded for.</param>
    /// <param name="outputBackend">Backend that hosts the returned hidden and logits tensors (the CPU backend).</param>
    internal DeepSeekV41GenerationModel(DeepSeekV41HostModel model, int maxTokens, IBackend outputBackend)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(outputBackend);
        _backend = outputBackend;
        _model = model;
        _maxTokens = maxTokens;
        Info = new GenerationModelInfo("DeepSeekV41", model.VocabSize, model.Dim, model.Layers, maxTokens);
    }

    /// <inheritdoc />
    public ISequenceState CreateSequenceState(SequenceStateOptions options)
    {
        if (options.MaxSequenceTokens > _maxTokens)
            throw new ArgumentOutOfRangeException(nameof(options), options.MaxSequenceTokens, $"The model was loaded for sequences of up to {_maxTokens} tokens.");
        return new DeepSeekV41GenerationState(_model, options.MaxSequenceTokens);
    }

    /// <inheritdoc />
    public Tensor Prefill(in PrefillChunk chunk, ISequenceState state) => Prefill(chunk, state, CancellationToken.None);

    /// <summary>Checks <paramref name="cancel"/> before running; a prefill is one forward pass and is not interruptible once started.</summary>
    public Tensor Prefill(in PrefillChunk chunk, ISequenceState state, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        DeepSeekV41GenerationState sequence = AsSequence(state);
        if (chunk.Embeds is not null) throw new NotSupportedException("The V4.1 reference model takes text tokens only; image embeddings are not supported.");
        ReadOnlySpan<int> ids = chunk.TokenIds.Span;
        if (ids.IsEmpty) throw new ArgumentException("Prefill needs at least one token.", nameof(chunk));
        if (chunk.PosStart != sequence.Length)
            throw new ArgumentException($"The chunk starts at position {chunk.PosStart} but the sequence holds {sequence.Length} tokens.", nameof(chunk));

        int dim = _model.Dim;
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
            DeepSeekV41GenerationState sequence = AsSequence(states[i]);
            if (!seen.Add(sequence)) throw new ArgumentException("A sequence state appears twice in one batch.", nameof(states));
            if (sequence.Length >= sequence.Capacity) throw new InvalidOperationException($"Sequence {i} is full ({sequence.Capacity} tokens).");
            if ((uint)tokenIds[i] >= (uint)_model.VocabSize) throw new ArgumentOutOfRangeException(nameof(tokenIds), tokenIds[i], "Token id is outside the vocabulary.");
        }
        int dim = _model.Dim;
        float[] hidden = new float[tokenIds.Length * dim];
        for (int i = 0; i < tokenIds.Length; i++) AsSequence(states[i]).Append(tokenIds.Slice(i, 1), hidden.AsSpan(i * dim, dim));
        return Hidden(hidden, tokenIds.Length);
    }

    /// <inheritdoc />
    public Tensor ProjectLogits(Tensor hidden, int rows)
    {
        ArgumentNullException.ThrowIfNull(hidden);
        int dim = _model.Dim, vocab = _model.VocabSize;
        if (hidden.DType != DType.F32 || hidden.ElementCount < (long)rows * dim)
            throw new ArgumentException($"hidden must be F32 with at least {rows} x {dim} values.", nameof(hidden));
        Tensor logits = new(new TensorShape(1, rows, vocab), DType.F32);
        ReadOnlySpan<float> source = hidden.AsReadOnlySpan<float>();
        Span<float> destination = logits.AsSpan<float>();
        for (int r = 0; r < rows; r++) _model.Logits(source.Slice(r * dim, dim)).CopyTo(destination.Slice(r * vocab, vocab));
        return logits;
    }

    /// <summary>Empty: the weights are host arrays and a checkpoint view, not engine tensors, so there is nothing for a residency planner to place.</summary>
    public IEnumerable<Tensor> EnumerateWeights(bool includeRedundantSplits) => [];

    /// <inheritdoc />
    public long EstimateSequenceBytes(int contextTokens) => _model.EstimateStateBytes(contextTokens);

    /// <summary>Free and total host memory as the runtime sees it; there is no device on this path.</summary>
    public CapacitySnapshot Capacity()
    {
        GCMemoryInfo info = GC.GetGCMemoryInfo();
        return new CapacitySnapshot(Math.Max(0, info.TotalAvailableMemoryBytes - info.MemoryLoadBytes), info.TotalAvailableMemoryBytes);
    }

    /// <inheritdoc />
    public void Dispose() => _loaded?.Dispose();

    private Tensor Hidden(ReadOnlySpan<float> values, int rows)
    {
        Tensor t = new(new TensorShape(1, rows, _model.Dim), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static DeepSeekV41GenerationState AsSequence(ISequenceState state) =>
        state as DeepSeekV41GenerationState ?? throw new ArgumentException($"{state.GetType().Name} is not a V4.1 sequence state.", nameof(state));
}
