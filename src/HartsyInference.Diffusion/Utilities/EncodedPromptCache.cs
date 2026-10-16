using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>The last encoded prompt of each slot (conditional, unconditional, …), keyed by its token ids and a caller
/// discriminator for anything else the encoding depends on (a padding mask's valid length). A text encoder is the
/// slowest thing a large-LM-conditioned pipeline runs per generation; re-sending the same prompt reuses the cached
/// conditioning instead. Emphasis weights are applied after encoding, so they are not part of the key. Owns the
/// cached tensors.</summary>
public sealed class EncodedPromptCache(int slots) : IDisposable
{
    private readonly int[]?[] _keys = new int[]?[slots > 0 ? slots : throw new ArgumentOutOfRangeException(nameof(slots))];
    private readonly int[] _discriminators = new int[slots];
    private readonly Tensor?[] _values = new Tensor?[slots];

    /// <summary>The cached encoding for <paramref name="ids"/> in <paramref name="slot"/>, if it is the last one stored there.</summary>
    public bool TryGet(int slot, ReadOnlySpan<int> ids, int discriminator, out Tensor? value)
    {
        value = _keys[slot] is int[] cached && _discriminators[slot] == discriminator && cached.AsSpan().SequenceEqual(ids)
            ? _values[slot] : null;
        return value is not null;
    }

    /// <summary>Takes ownership of <paramref name="value"/> as <paramref name="slot"/>'s encoding of <paramref name="ids"/>,
    /// disposing whatever the slot held.</summary>
    public void Store(int slot, ReadOnlySpan<int> ids, int discriminator, Tensor value)
    {
        ArgumentNullException.ThrowIfNull(value);
        if (!ReferenceEquals(_values[slot], value))
            _values[slot]?.Dispose();
        _keys[slot] = ids.ToArray();
        _discriminators[slot] = discriminator;
        _values[slot] = value;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        for (int i = 0; i < _values.Length; i++)
        {
            _values[i]?.Dispose();
            _values[i] = null;
            _keys[i] = null;
        }
    }
}
