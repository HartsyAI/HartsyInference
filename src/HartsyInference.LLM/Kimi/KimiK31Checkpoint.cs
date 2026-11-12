using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.LLM.Kimi;

/// <summary>A Kimi K3.1 decoder loaded from a checkpoint directory: its parsed config, the model, and the shard set the model's weights
/// are views into. Dispose it to release the model and the mappings. The backends are borrowed; their owner disposes them.</summary>
public sealed class KimiK31Checkpoint : IDisposable
{
    private KimiK31CheckpointWeights? _weights;
    private KimiK31Model? _model;
    private IReadOnlyList<Tensor> _owned = [];

    private KimiK31Checkpoint(KimiK31Config config, KimiK31CheckpointWeights weights, KimiK31Model model)
    {
        Config = config;
        _weights = weights;
        _model = model;
    }

    /// <summary>The config the checkpoint's <c>config.json</c> describes.</summary>
    public KimiK31Config Config { get; }

    /// <summary>The decoder, ready to run.</summary>
    public KimiK31Model Model => _model ?? throw new ObjectDisposedException(nameof(KimiK31Checkpoint));

    /// <summary>Reads <c>config.json</c> and the shards in <paramref name="directory"/>, and builds the decoder on <paramref name="backend"/>.
    /// <paramref name="maxPositions"/> sizes the positional tables the attention layers read.</summary>
    /// <exception cref="System.IO.InvalidDataException">The config or a tensor does not match the Kimi K3.1 decoder.</exception>
    /// <exception cref="NotSupportedException">The decoder cannot run a tensor the checkpoint stores in a format it does not support.</exception>
    public static KimiK31Checkpoint Load(string directory, IBackend backend, int maxPositions)
    {
        ArgumentNullException.ThrowIfNull(backend);
        return Load(directory, [backend], null, maxPositions);
    }

    /// <summary>Builds the decoder sharded over <paramref name="rankBackends"/>, one backend per rank. Each rank loads only its own share of
    /// the checkpoint (its heads, vocabulary rows, FFN columns and routed experts), with dense weights in the dtype its backend runs.</summary>
    /// <param name="rankBackends">Rank-ordered backends; one backend loads the whole decoder.</param>
    /// <param name="comm">The collective over the ranks; required with two or more backends, unused with one.</param>
    /// <exception cref="ArgumentException">The config does not split across the ranks, or the communicator does not span them.</exception>
    /// <exception cref="System.IO.InvalidDataException">The config or a tensor does not match the Kimi K3.1 decoder.</exception>
    public static KimiK31Checkpoint Load(string directory, IReadOnlyList<IBackend> rankBackends, ICollectiveComm? comm, int maxPositions)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        ArgumentNullException.ThrowIfNull(rankBackends);
        if (rankBackends.Count == 0) throw new ArgumentException("A Kimi K3.1 load needs at least one rank backend.", nameof(rankBackends));
        KimiK31Config config = KimiK31HfConfig.Parse(File.ReadAllText(Path.Combine(directory, "config.json")));
        KimiK31CheckpointWeights weights = KimiK31CheckpointWeights.Open(directory);
        List<KimiK31Weights> loaded = new(rankBackends.Count);
        try
        {
            int ranks = rankBackends.Count;
            for (int r = 0; r < ranks; r++)
            {
                IBackend backend = rankBackends[r];
                loaded.Add(KimiK31WeightLoader.Load(config, weights, new KimiK31Shard(r, ranks), (t, key) => DenseWeightCasts.ForBackend(t, backend, key)));
            }
            KimiK31Model model = ranks == 1
                ? new KimiK31Model(rankBackends[0], config, loaded[0], maxPositions)
                : new KimiK31Model(new TpPlacement(rankBackends, comm ?? throw new ArgumentException("A sharded Kimi K3.1 load needs a communicator.", nameof(comm))),
                    config, loaded, maxPositions);
            KimiK31Checkpoint checkpoint = new(config, weights, model);
            checkpoint._owned = loaded.SelectMany(w => w.Owned).ToList();
            return checkpoint;
        }
        catch
        {
            // The model was never handed out, so the tensors it would have owned are released here.
            foreach (KimiK31Weights rank in loaded) foreach (Tensor t in rank.Owned) t.Dispose();
            weights.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _model?.Dispose();
        _model = null;
        foreach (Tensor t in _owned) t.Dispose();
        _owned = [];
        _weights?.Dispose();
        _weights = null;
    }
}
