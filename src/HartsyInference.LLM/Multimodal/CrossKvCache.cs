using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Multimodal;

/// <summary>The projected, normalized, rotated keys and values of one fixed vision sequence, per gated cross-attention layer.
/// The vision features do not change between the decode steps of a request, so each layer projects them once and later steps reuse
/// the result instead of re-projecting every vision token at every token.</summary>
/// <remarks>One cache serves one vision sequence and its rotary tables. The caller creates it per request, passes it to every forward
/// over that sequence, and disposes it when the request ends. Keys and values are head-major <c>[1, heads, L, head_dim]</c> F32.</remarks>
public sealed class CrossKvCache(int layers) : IDisposable
{
    private readonly (Tensor K, Tensor V)?[] _entries = new (Tensor K, Tensor V)?[layers];
    private int _disposed;

    /// <summary>The cached keys and values of <paramref name="layer"/>, or null before that layer has projected them.</summary>
    public (Tensor K, Tensor V)? Get(int layer) => _entries[layer];

    /// <summary>Stores the keys and values of <paramref name="layer"/>. The cache takes ownership of both tensors.</summary>
    /// <exception cref="InvalidOperationException">The layer already holds keys and values.</exception>
    public void Set(int layer, Tensor k, Tensor v)
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(CrossKvCache));
        if (_entries[layer] is not null) throw new InvalidOperationException($"Cross-attention layer {layer} is already projected.");
        _entries[layer] = (k, v);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        for (int i = 0; i < _entries.Length; i++)
        {
            if (_entries[i] is not { } e) continue;
            e.K.Dispose();
            e.V.Dispose();
            _entries[i] = null;
        }
    }
}
