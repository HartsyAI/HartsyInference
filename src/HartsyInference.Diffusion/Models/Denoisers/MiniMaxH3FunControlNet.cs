using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>One validated, reusable MiniMax-H3 Fun ControlNet-Union weight branch.</summary>
public sealed class MiniMaxH3FunControlNet(MiniMaxH3FunControlConfig config) : IDisposable
{
    private readonly Dictionary<string, Tensor> _weights = new Dictionary<string, Tensor>(StringComparer.Ordinal);
    private bool _disposed;

    /// <summary>Header-derived branch geometry.</summary>
    public MiniMaxH3FunControlConfig Config { get; } = config;

    /// <summary>Takes ownership of one completely converted control checkpoint.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(weights);
        MiniMaxH3FunControlConfig detected = MiniMaxH3FunControlConfig.Detect(weights);
        if (detected.HiddenSize != Config.HiddenSize || detected.NumBlocks != Config.NumBlocks
            || detected.NumAttentionHeads != Config.NumAttentionHeads
            || detected.AttentionHeadDim != Config.AttentionHeadDim
            || detected.FfnHiddenSize != Config.FfnHiddenSize || detected.TimeEmbedDim != Config.TimeEmbedDim
            || detected.ControlInputChannels != Config.ControlInputChannels
            || !detected.InjectionLayers.SequenceEqual(Config.InjectionLayers))
        {
            throw new ArgumentException(
                $"MiniMax-H3 Fun ControlNet config changed between detection and load: {Config} != {detected}.",
                nameof(weights));
        }
        foreach (KeyValuePair<string, Tensor> pair in weights)
        {
            _weights.Add(pair.Key, pair.Value);
        }
    }

    /// <summary>All branch weights, including projections and all five control blocks.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _weights.Values;
    }

    /// <summary>Returns one required converted tensor for the transformer's shared block runner.</summary>
    internal Tensor Require(string key)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _weights.TryGetValue(key, out Tensor? tensor) ? tensor
            : throw new KeyNotFoundException($"MiniMax-H3 Fun ControlNet weight '{key}' is missing.");
    }

    /// <summary>Returns one optional converted tensor for the transformer's shared block runner.</summary>
    internal Tensor? Optional(string key)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _weights.TryGetValue(key, out Tensor? tensor) ? tensor : null;
    }

    /// <summary>Releases owned converted tensors; mmap-backed tensors remain governed by their loader.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        foreach (Tensor tensor in _weights.Values)
        {
            tensor.Dispose();
        }
        _weights.Clear();
    }
}
