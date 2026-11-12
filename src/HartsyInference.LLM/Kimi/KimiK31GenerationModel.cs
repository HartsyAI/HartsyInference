using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Generation;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Kimi;

/// <summary>Adapts a loaded Kimi K3.1 decoder to <see cref="IGenerationModel"/> so the shared pipeline can drive it.</summary>
/// <remarks>Owns the checkpoint. Text only, one token per decode step, no batched decode. The decoder runs on the backend it was built on, so
/// the hidden and logits tensors come back on the host. Not thread-safe: run one sequence at a time.</remarks>
public sealed class KimiK31GenerationModel : IGenerationModel
{
    private readonly KimiK31Checkpoint _checkpoint;
    private readonly int _maxTokens;

    /// <param name="checkpoint">The loaded checkpoint; this adapter takes ownership.</param>
    /// <param name="outputBackend">Backend the decoder was built on.</param>
    /// <param name="maxTokens">The longest sequence the decoder was loaded for.</param>
    public KimiK31GenerationModel(KimiK31Checkpoint checkpoint, IBackend outputBackend, int maxTokens)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        ArgumentNullException.ThrowIfNull(outputBackend);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxTokens, 1);
        _checkpoint = checkpoint;
        OutputBackend = outputBackend;
        _maxTokens = maxTokens;
        Info = new GenerationModelInfo("Kimi K3.1", checkpoint.Config.Vocab, checkpoint.Config.Hidden, checkpoint.Config.Layers, maxTokens);
    }

    private KimiK31Model Model => _checkpoint.Model;

    private KimiK31Config Config => _checkpoint.Config;

    /// <summary>The decoder, for the image route's embedding splice.</summary>
    internal KimiK31Model Decoder => Model;

    /// <summary>The config the decoder was built from.</summary>
    internal KimiK31Config DecoderConfig => Config;

    /// <summary>The longest sequence the decoder was loaded for.</summary>
    internal int MaxSequenceTokens => _maxTokens;

    /// <inheritdoc />
    public GenerationModelInfo Info { get; }

    /// <inheritdoc />
    public GenerationCapabilities Capabilities { get; } = new();

    /// <inheritdoc />
    public IBackend OutputBackend { get; }

    /// <inheritdoc />
    public ISequenceState CreateSequenceState(SequenceStateOptions options)
    {
        if (options.MaxSequenceTokens > _maxTokens)
            throw new ArgumentOutOfRangeException(nameof(options), options.MaxSequenceTokens, $"The model was loaded for sequences of up to {_maxTokens} tokens.");
        int capacity = options.MaxSequenceTokens;
        SequenceCache cache = new(new KimiK31State(Config, capacity));
        int dim = Config.Hidden;
        return new HostReplaySequenceState(capacity, dim,
            (ids, hidden) => ForwardTokens(ids, hidden, cache),
            () => cache.Reset(Config, capacity),
            cache.Dispose);
    }

    /// <inheritdoc />
    public Tensor Prefill(in PrefillChunk chunk, ISequenceState state) => Prefill(chunk, state, CancellationToken.None);

    /// <summary>Checks <paramref name="cancel"/> before running; a prefill is one forward pass and is not interruptible once started.</summary>
    public Tensor Prefill(in PrefillChunk chunk, ISequenceState state, CancellationToken cancel)
    {
        cancel.ThrowIfCancellationRequested();
        HostReplaySequenceState sequence = AsSequence(state);
        if (chunk.Embeds is not null) throw new NotSupportedException("The Kimi K3.1 decoder takes text tokens only; image embeddings are not supported.");
        ReadOnlySpan<int> ids = chunk.TokenIds.Span;
        if (ids.IsEmpty) throw new ArgumentException("Prefill needs at least one token.", nameof(chunk));
        if (chunk.PosStart != sequence.Length)
            throw new ArgumentException($"The chunk starts at position {chunk.PosStart} but the sequence holds {sequence.Length} tokens.", nameof(chunk));
        // checked here, before the sequence moves: a failure inside the forward resets the sequence
        foreach (int id in ids)
            if ((uint)id >= (uint)Config.Vocab) throw new ArgumentOutOfRangeException(nameof(chunk), id, "Token id is outside the vocabulary.");

        int dim = Config.Hidden;
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
            if ((uint)tokenIds[i] >= (uint)Config.Vocab) throw new ArgumentOutOfRangeException(nameof(tokenIds), tokenIds[i], "Token id is outside the vocabulary.");
        }
        int dim = Config.Hidden;
        float[] hidden = new float[tokenIds.Length * dim];
        for (int i = 0; i < tokenIds.Length; i++) AsSequence(states[i]).Append(tokenIds.Slice(i, 1), hidden.AsSpan(i * dim, dim));
        return Hidden(hidden, tokenIds.Length);
    }

    /// <inheritdoc />
    public Tensor ProjectLogits(Tensor hidden, int rows)
    {
        ArgumentNullException.ThrowIfNull(hidden);
        int dim = Config.Hidden, vocab = Config.Vocab;
        if (hidden.DType != DType.F32 || hidden.ElementCount < (long)rows * dim)
            throw new ArgumentException($"hidden must be F32 with at least {rows} x {dim} values.", nameof(hidden));
        ReadOnlySpan<float> source = hidden.AsReadOnlySpan<float>();
        Tensor logits = new(new TensorShape(1, rows, vocab), DType.F32);
        Span<float> target = logits.AsSpan<float>();
        for (int r = 0; r < rows; r++)
        {
            float[] row = Model.ProjectHidden(source.Slice(r * dim, dim).ToArray());
            row.AsSpan().CopyTo(target.Slice(r * vocab, vocab));
        }
        return logits;
    }

    /// <summary>Empty: the decoder's weights are owned by the checkpoint and are not engine tensors on the residency path.</summary>
    public IEnumerable<Tensor> EnumerateWeights(bool includeRedundantSplits) => [];

    /// <summary>A rough estimate: one MLA latent, rope and hidden row per context token and layer. KDA recurrent state is per sequence and not counted.</summary>
    public long EstimateSequenceBytes(int contextTokens)
    {
        long rowFloats = (long)Config.KvLoraRank + Config.QkRopeDim + Config.Hidden;
        return (long)contextTokens * Config.Layers * rowFloats * sizeof(float);
    }

    /// <summary>Free and total host memory as the runtime sees it; the decoder runs on the host or on the backend the checkpoint was loaded to.</summary>
    public CapacitySnapshot Capacity()
    {
        GCMemoryInfo info = GC.GetGCMemoryInfo();
        return new CapacitySnapshot(Math.Max(0, info.TotalAvailableMemoryBytes - info.MemoryLoadBytes), info.TotalAvailableMemoryBytes);
    }

    /// <inheritdoc />
    public void Dispose() => _checkpoint.Dispose();

    private void ForwardTokens(ReadOnlySpan<int> tokenIds, Span<float> hidden, SequenceCache cache)
    {
        int dim = Config.Hidden;
        for (int i = 0; i < tokenIds.Length; i++)
        {
            float[] row = Model.ForwardTokenHidden(tokenIds[i], cache.State);
            row.AsSpan().CopyTo(hidden.Slice(i * dim, dim));
        }
    }

    private Tensor Hidden(ReadOnlySpan<float> values, int rows)
    {
        Tensor t = new(new TensorShape(1, rows, Config.Hidden), DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    private static HostReplaySequenceState AsSequence(ISequenceState state) =>
        state as HostReplaySequenceState ?? throw new ArgumentException($"{state.GetType().Name} is not a Kimi K3.1 sequence state.", nameof(state));

    // Holds the decoder state of one sequence, replaced when the sequence resets.
    private sealed class SequenceCache(KimiK31State state) : IDisposable
    {
        private KimiK31State _state = state;
        private bool _disposed;

        public KimiK31State State => _disposed ? throw new ObjectDisposedException(nameof(SequenceCache)) : _state;

        public void Reset(KimiK31Config config, int capacity)
        {
            _state.Dispose();
            _state = new KimiK31State(config, capacity);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _state.Dispose();
        }
    }
}
