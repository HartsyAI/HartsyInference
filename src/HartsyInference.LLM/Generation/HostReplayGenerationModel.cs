using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Generation;

/// <summary>The <see cref="IGenerationModel"/> shell of a host-reference decoder whose sequences replay through <see cref="HostReplaySequenceState"/>; a family supplies its state, logits and estimate.</summary>
/// <remarks>Text only, one token per decode step, no speculation. A prefill must start at the sequence's committed length. Hidden and logits tensors are host tensors whatever the backend.
/// Not thread-safe: run one sequence at a time.</remarks>
public abstract class HostReplayGenerationModel : IGenerationModel
{
    private readonly string _label;
    private readonly string _embedsRefusal;
    private readonly int _vocab;
    private readonly int _hidden;

    /// <summary>Validates <paramref name="outputBackend"/> and <paramref name="maxTokens"/>; <paramref name="label"/> names the family in a refused sequence state.</summary>
    /// <param name="name">The model name <see cref="Info"/> reports.</param>
    /// <param name="vocab">Vocabulary size: token ids at or above it are refused.</param>
    /// <param name="hidden">Hidden width of one row.</param>
    /// <param name="layers">Decoder layer count, for <see cref="Info"/>.</param>
    /// <param name="outputBackend">Backend the model runs on.</param>
    /// <param name="maxTokens">The longest sequence the model was loaded for.</param>
    /// <param name="label">The family's name in messages, such as <c>Maxwell</c>.</param>
    /// <param name="embedsRefusal">The message a prefill that carries embeddings is refused with.</param>
    /// <param name="batchDecode">Whether <see cref="DecodeBatch"/> takes more than one sequence per call.</param>
    protected HostReplayGenerationModel(string name, int vocab, int hidden, int layers, IBackend outputBackend, int maxTokens, string label,
        string embedsRefusal, bool batchDecode)
    {
        ArgumentNullException.ThrowIfNull(outputBackend);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxTokens, 1);
        _label = label;
        _embedsRefusal = embedsRefusal;
        _vocab = vocab;
        _hidden = hidden;
        OutputBackend = outputBackend;
        MaxTokens = maxTokens;
        Info = new GenerationModelInfo(name, vocab, hidden, layers, maxTokens);
        Capabilities = new GenerationCapabilities { SupportsBatchDecode = batchDecode };
    }

    /// <summary>The longest sequence the model was loaded for.</summary>
    public int MaxTokens { get; }

    /// <inheritdoc />
    public GenerationModelInfo Info { get; }

    /// <inheritdoc />
    public GenerationCapabilities Capabilities { get; }

    /// <inheritdoc />
    public IBackend OutputBackend { get; }

    /// <inheritdoc />
    public ISequenceState CreateSequenceState(SequenceStateOptions options)
    {
        if (options.MaxSequenceTokens > MaxTokens)
            throw new ArgumentOutOfRangeException(nameof(options), options.MaxSequenceTokens, $"The model was loaded for sequences of up to {MaxTokens} tokens.");
        return CreateReplayState(options.MaxSequenceTokens);
    }

    /// <inheritdoc />
    public Tensor Prefill(in PrefillChunk chunk, ISequenceState state) => Prefill(chunk, state, CancellationToken.None);

    /// <summary>Checks <paramref name="cancel"/> before running; a prefill is one forward pass and is not interruptible once started.</summary>
    public Tensor Prefill(in PrefillChunk chunk, ISequenceState state, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        HostReplaySequenceState sequence = AsSequence(state);
        if (chunk.Embeds is not null) throw new NotSupportedException(_embedsRefusal);
        ReadOnlySpan<int> ids = chunk.TokenIds.Span;
        if (ids.IsEmpty) throw new ArgumentException("Prefill needs at least one token.", nameof(chunk));
        if (chunk.PosStart != sequence.Length)
            throw new ArgumentException($"The chunk starts at position {chunk.PosStart} but the sequence holds {sequence.Length} tokens.", nameof(chunk));
        // Checked before the sequence moves: a failure inside the forward resets the sequence.
        foreach (int id in ids)
            if ((uint)id >= (uint)_vocab) throw new ArgumentOutOfRangeException(nameof(chunk), id, "Token id is outside the vocabulary.");

        float[] hidden = new float[ids.Length * _hidden];
        sequence.Append(ids, hidden);
        int rows = chunk.LastRowOnly ? 1 : ids.Length;
        return Hidden(hidden.AsSpan((ids.Length - rows) * _hidden, rows * _hidden), rows);
    }

    /// <inheritdoc />
    public Tensor DecodeBatch(ReadOnlySpan<int> tokenIds, ISequenceState[] states)
    {
        ArgumentNullException.ThrowIfNull(states);
        if (states.Length != tokenIds.Length) throw new ArgumentException($"{tokenIds.Length} tokens for {states.Length} sequence states.");
        // Refuse the whole batch before any sequence moves, as the single-sequence path does.
        HashSet<ISequenceState> seen = new(ReferenceEqualityComparer.Instance);
        for (int i = 0; i < tokenIds.Length; i++)
        {
            HostReplaySequenceState sequence = AsSequence(states[i]);
            if (!seen.Add(sequence)) throw new ArgumentException("A sequence state appears twice in one batch.", nameof(states));
            if (sequence.Length >= sequence.Capacity) throw new InvalidOperationException($"Sequence {i} is full ({sequence.Capacity} tokens).");
            if ((uint)tokenIds[i] >= (uint)_vocab) throw new ArgumentOutOfRangeException(nameof(tokenIds), tokenIds[i], "Token id is outside the vocabulary.");
        }
        float[] hidden = new float[tokenIds.Length * _hidden];
        for (int i = 0; i < tokenIds.Length; i++) AsSequence(states[i]).Append(tokenIds.Slice(i, 1), hidden.AsSpan(i * _hidden, _hidden));
        return Hidden(hidden, tokenIds.Length);
    }

    /// <inheritdoc />
    public Tensor ProjectLogits(Tensor hidden, int rows)
    {
        ArgumentNullException.ThrowIfNull(hidden);
        if (hidden.DType != DType.F32 || hidden.ElementCount < (long)rows * _hidden)
            throw new ArgumentException($"hidden must be F32 with at least {rows} x {_hidden} values.", nameof(hidden));
        Tensor logits = new(new TensorShape(1, rows, _vocab), DType.F32);
        try
        {
            ProjectRows(hidden.AsReadOnlySpan<float>().Slice(0, rows * _hidden), rows, logits.AsSpan<float>());
            return logits;
        }
        catch
        {
            logits.Dispose();
            throw;
        }
    }

    /// <summary>Empty: the weights are host arrays and checkpoint views, not engine tensors, so there is nothing for a residency planner to place.</summary>
    public IEnumerable<Tensor> EnumerateWeights(bool includeRedundantSplits) => [];

    /// <inheritdoc />
    public abstract long EstimateSequenceBytes(int contextTokens);

    /// <summary>Free and total host memory as the runtime sees it: the sequence states live on the host.</summary>
    public CapacitySnapshot Capacity()
    {
        GCMemoryInfo info = GC.GetGCMemoryInfo();
        return new CapacitySnapshot(Math.Max(0, info.TotalAvailableMemoryBytes - info.MemoryLoadBytes), info.TotalAvailableMemoryBytes);
    }

    /// <inheritdoc />
    public void Dispose() => DisposeModel();

    /// <summary>A fresh sequence of up to <paramref name="capacity"/> tokens; its forward writes one hidden row per token id.</summary>
    protected abstract HostReplaySequenceState CreateReplayState(int capacity);

    /// <summary>Writes the logits of <paramref name="rows"/> hidden rows into <paramref name="logits"/>, one vocabulary-wide row each.</summary>
    protected abstract void ProjectRows(ReadOnlySpan<float> hidden, int rows, Span<float> logits);

    /// <summary>Releases the loaded model this adapter owns.</summary>
    protected abstract void DisposeModel();

    /// <summary>Throws the <see cref="ArgumentNullException"/> a constructor would, for a model argument read before the base constructor runs.</summary>
    protected static T Required<T>(T value, string paramName) where T : class
    {
        ArgumentNullException.ThrowIfNull(value, paramName);
        return value;
    }

    private Tensor Hidden(ReadOnlySpan<float> values, int rows)
    {
        Tensor t = new(new TensorShape(1, rows, _hidden), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private HostReplaySequenceState AsSequence(ISequenceState state) =>
        state as HostReplaySequenceState ?? throw new ArgumentException($"{state.GetType().Name} is not a {_label} sequence state.", nameof(state));
}
