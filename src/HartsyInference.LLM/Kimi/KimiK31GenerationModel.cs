using HartsyInference.Core.Backends;
using HartsyInference.LLM.Generation;

namespace HartsyInference.LLM.Kimi;

/// <summary>Adapts a loaded Kimi K3.1 decoder to <see cref="IGenerationModel"/> so the shared pipeline can drive it.</summary>
/// <remarks>Owns the checkpoint. Text only, one token per decode step, no batched decode. The decoder runs on the backend it was built on, so
/// the hidden and logits tensors come back on the host. Not thread-safe: run one sequence at a time.</remarks>
public sealed class KimiK31GenerationModel : HostReplayGenerationModel
{
    // TODO(kimi): the CUDA and Vulkan decoder is not run on GPU through this adapter. Finish it by running generation on a GPU backend and checking its
    // tokens against the CPU run.
    // TODO(kimi): no memory guard exists, and Capacity() reports host memory even when the decoder runs on a GPU. Finish it by reporting device
    // memory for a GPU-built decoder and checking EstimateSequenceBytes against it before a request.
    private readonly KimiK31Checkpoint _checkpoint;

    /// <param name="checkpoint">The loaded checkpoint; this adapter takes ownership.</param>
    /// <param name="outputBackend">Backend the decoder was built on.</param>
    /// <param name="maxTokens">The longest sequence the decoder was loaded for.</param>
    public KimiK31GenerationModel(KimiK31Checkpoint checkpoint, IBackend outputBackend, int maxTokens)
        : base("Kimi K3.1", Required(checkpoint, nameof(checkpoint)).Config.Vocab, checkpoint.Config.Hidden, checkpoint.Config.Layers, outputBackend,
            maxTokens, "Kimi K3.1", "The Kimi K3.1 decoder takes text tokens only; image embeddings are not supported.", batchDecode: false)
    {
        _checkpoint = checkpoint;
    }

    private KimiK31Model Model => _checkpoint.Model;

    private KimiK31Config Config => _checkpoint.Config;

    /// <summary>The decoder, for the image route's embedding splice.</summary>
    internal KimiK31Model Decoder => Model;

    /// <summary>The config the decoder was built from.</summary>
    internal KimiK31Config DecoderConfig => Config;

    /// <summary>The longest sequence the decoder was loaded for.</summary>
    internal int MaxSequenceTokens => MaxTokens;

    /// <summary>A rough estimate: one MLA latent, rope and hidden row per context token and layer. KDA recurrent state is per sequence and not counted.</summary>
    public override long EstimateSequenceBytes(int contextTokens)
    {
        long rowFloats = (long)Config.KvLoraRank + Config.QkRopeDim + Config.Hidden;
        return (long)contextTokens * Config.Layers * rowFloats * sizeof(float);
    }

    /// <inheritdoc />
    protected override HostReplaySequenceState CreateReplayState(int capacity)
    {
        SequenceCache cache = new(new KimiK31State(Config, capacity, Model.Ranks));
        return new HostReplaySequenceState(capacity, Config.Hidden,
            (ids, hidden) => ForwardTokens(ids, hidden, cache),
            () => cache.Reset(Config, capacity, Model.Ranks),
            cache.Dispose);
    }

    /// <inheritdoc />
    protected override void ProjectRows(ReadOnlySpan<float> hidden, int rows, Span<float> logits)
    {
        int dim = Config.Hidden, vocab = Config.Vocab;
        for (int r = 0; r < rows; r++)
        {
            float[] row = Model.ProjectHidden(hidden.Slice(r * dim, dim).ToArray());
            row.AsSpan().CopyTo(logits.Slice(r * vocab, vocab));
        }
    }

    /// <inheritdoc />
    protected override void DisposeModel() => _checkpoint.Dispose();

    private void ForwardTokens(ReadOnlySpan<int> tokenIds, Span<float> hidden, SequenceCache cache)
    {
        int dim = Config.Hidden;
        for (int i = 0; i < tokenIds.Length; i++)
        {
            float[] row = Model.ForwardTokenHidden(tokenIds[i], cache.State);
            row.AsSpan().CopyTo(hidden.Slice(i * dim, dim));
        }
    }

    // Holds the decoder state of one sequence, replaced when the sequence resets.
    private sealed class SequenceCache(KimiK31State state) : IDisposable
    {
        private KimiK31State _state = state;
        private bool _disposed;

        public KimiK31State State => _disposed ? throw new ObjectDisposedException(nameof(SequenceCache)) : _state;

        public void Reset(KimiK31Config config, int capacity, int ranks)
        {
            _state.Dispose();
            _state = new KimiK31State(config, capacity, ranks);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _state.Dispose();
        }
    }
}
