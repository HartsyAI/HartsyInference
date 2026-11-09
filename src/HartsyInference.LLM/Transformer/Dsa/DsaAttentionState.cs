using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer.Dsa;

/// <summary>One sequence's cache for a DSA layer: a latent row and an indexer key per position, F32 on the host.</summary>
public sealed class DsaAttentionState : IDisposable
{
    private Tensor? _latent;
    private Tensor? _indexKeys;

    /// <param name="capacity">Positions the cache holds.</param>
    /// <param name="settings">The layer the state is for.</param>
    public DsaAttentionState(int capacity, DsaAttentionSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        Capacity = capacity;
        _latent = new Tensor(new TensorShape(capacity, settings.LatentDim), DType.F32);
        _indexKeys = new Tensor(new TensorShape(capacity, settings.Indexer.HeadDim), DType.F32);
    }

    /// <summary>Positions the cache holds.</summary>
    public int Capacity { get; }

    /// <summary>Positions written so far.</summary>
    public int Length { get; internal set; }

    /// <summary>Latent rows <c>[Capacity, KvLoraRank + QkRopeDim]</c>: the normalized latent then the rotated shared key.</summary>
    public Tensor Latent => _latent ?? throw new ObjectDisposedException(nameof(DsaAttentionState));

    /// <summary>Indexer keys <c>[Capacity, Indexer.HeadDim]</c>, after RoPE, Hadamard and the key round trip.</summary>
    public Tensor IndexKeys => _indexKeys ?? throw new ObjectDisposedException(nameof(DsaAttentionState));

    /// <summary>Starts a new sequence; the rows are overwritten as positions arrive.</summary>
    public void Reset() => Length = 0;

    /// <inheritdoc />
    public void Dispose()
    {
        Interlocked.Exchange(ref _latent, null)?.Dispose();
        Interlocked.Exchange(ref _indexKeys, null)?.Dispose();
    }
}
