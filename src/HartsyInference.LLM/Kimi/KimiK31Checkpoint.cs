using HartsyInference.Core.Backends;

namespace HartsyInference.LLM.Kimi;

/// <summary>A Kimi K3.1 decoder loaded from a checkpoint directory: its parsed config, the model, and the shard set the model's weights
/// are views into. Dispose it to release the model and the mappings.</summary>
public sealed class KimiK31Checkpoint : IDisposable
{
    private KimiK31CheckpointWeights? _weights;
    private KimiK31Model? _model;

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
        ArgumentException.ThrowIfNullOrEmpty(directory);
        KimiK31Config config = KimiK31HfConfig.Parse(File.ReadAllText(Path.Combine(directory, "config.json")));
        KimiK31CheckpointWeights weights = KimiK31CheckpointWeights.Open(directory);
        try
        {
            KimiK31Weights loaded = KimiK31WeightLoader.Load(config, weights);
            KimiK31Model model = new(backend, config, loaded, maxPositions);
            return new KimiK31Checkpoint(config, weights, model);
        }
        catch
        {
            weights.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _model?.Dispose();
        _model = null;
        _weights?.Dispose();
        _weights = null;
    }
}
