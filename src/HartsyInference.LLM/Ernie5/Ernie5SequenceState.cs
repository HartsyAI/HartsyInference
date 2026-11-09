using HartsyInference.LLM.Transformer.Compression;
using HartsyInference.LLM.Transformer.Dsa;

namespace HartsyInference.LLM.Ernie5;

/// <summary>Per-sequence caches of every ERNIE-5.5 decoder layer: an <see cref="HcaAttentionState"/> on HCA layers and a
/// <see cref="DsaAttentionState"/> on DSA layers. The FFNs and hyper-connections keep no state.</summary>
public sealed class Ernie5SequenceState : IDisposable
{
    private readonly HcaAttentionState?[] _hca;
    private readonly DsaAttentionState?[] _dsa;
    private int _disposed;

    /// <param name="config">The model's config.</param>
    /// <param name="capacity">Longest sequence this state will hold.</param>
    public Ernie5SequenceState(Ernie5Config config, int capacity)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        Capacity = capacity;
        _hca = new HcaAttentionState?[config.NumHiddenLayers];
        _dsa = new DsaAttentionState?[config.NumHiddenLayers];
        try
        {
            for (int i = 0; i < config.NumHiddenLayers; i++)
            {
                if (config.IsDsaLayer(i)) _dsa[i] = new DsaAttentionState(capacity, Ernie5Layer.DsaSettings(config));
                else _hca[i] = new HcaAttentionState(capacity, Ernie5Layer.HcaSettings(config, i));
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>Longest sequence the state holds.</summary>
    public int Capacity { get; }

    /// <summary>Tokens committed so far.</summary>
    public int Length { get; internal set; }

    /// <summary>Set when a forward failed partway through its layers, so the caches no longer match <see cref="Length"/>.</summary>
    internal bool Faulted { get; private set; }

    /// <summary>The HCA cache of layer <paramref name="layer"/>.</summary>
    internal HcaAttentionState Hca(int layer) => _hca[layer] ?? throw new InvalidOperationException($"Layer {layer} is not an HCA layer.");

    /// <summary>The DSA cache of layer <paramref name="layer"/>.</summary>
    internal DsaAttentionState Dsa(int layer) => _dsa[layer] ?? throw new InvalidOperationException($"Layer {layer} is not a DSA layer.");

    /// <summary>Marks the caches as out of step with <see cref="Length"/>; <see cref="Reset"/> clears it.</summary>
    internal void MarkFaulted() => Faulted = true;

    /// <summary>Forgets the sequence; the next forward starts at position 0.</summary>
    public void Reset()
    {
        Length = 0;
        Faulted = false;
        foreach (HcaAttentionState? s in _hca) s?.Reset();
        foreach (DsaAttentionState? s in _dsa) s?.Reset();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (DsaAttentionState? s in _dsa) s?.Dispose();
    }
}
