using HartsyInference.LLM.Transformer.Dsa;
using HartsyInference.LLM.Transformer.Kda;

namespace HartsyInference.LLM.Glm5Next;

/// <summary>One sequence's cache across the decoder: a conv history and delta-rule state per KDA layer, a latent row and indexer key row per position
/// per DSA layer. F32 on the host.</summary>
public sealed class Glm5NextState : IDisposable
{
    private readonly KdaState?[] _kda;
    private readonly DsaAttentionState?[] _dsa;

    /// <param name="config">The decoder the state is for.</param>
    /// <param name="capacity">Positions the DSA caches hold.</param>
    public Glm5NextState(Glm5NextConfig config, int capacity)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        Capacity = capacity;
        _kda = new KdaState?[config.Layers];
        _dsa = new DsaAttentionState?[config.Layers];
        for (int i = 0; i < config.Layers; i++)
        {
            if (config.LayerKinds[i] == Glm5NextAttentionKind.Kda) _kda[i] = new KdaState(Glm5NextLayer.KdaSettingsFor(config));
            else _dsa[i] = new DsaAttentionState(capacity, Glm5NextLayer.MlaSettingsFor(config), 2 * config.IndexHeadDim);
        }
    }

    /// <summary>Positions the DSA caches hold.</summary>
    public int Capacity { get; }

    /// <summary>Positions processed so far.</summary>
    public int Length { get; internal set; }

    internal KdaState Kda(int layer) => _kda[layer] ?? throw new InvalidOperationException($"Layer {layer} is not a KDA layer.");

    internal DsaAttentionState Dsa(int layer) => _dsa[layer] ?? throw new InvalidOperationException($"Layer {layer} is not a DSA layer.");

    /// <summary>Starts a new sequence.</summary>
    public void Reset()
    {
        Length = 0;
        foreach (KdaState? k in _kda)
        {
            k?.History.AsSpan<float>().Clear();
            k?.Recurrent.AsSpan<float>().Clear();
        }
        foreach (DsaAttentionState? d in _dsa) d?.Reset();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        foreach (KdaState? k in _kda) k?.Dispose();
        foreach (DsaAttentionState? d in _dsa) d?.Dispose();
    }
}
