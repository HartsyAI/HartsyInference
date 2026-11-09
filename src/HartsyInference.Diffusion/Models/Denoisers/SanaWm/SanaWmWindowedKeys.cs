using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Streaming;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>The keys and values one softmax attention call of a chunk-causal stream reads: the window kept from earlier
/// chunks, then the chunk's own. Without a window, or while it is empty (a stream's first chunk), these are the chunk's own
/// tensors, borrowed; otherwise they are joined copies this scope owns.</summary>
internal readonly struct SanaWmWindowedKeys : IDisposable
{
    private const int TokenAxis = 2;
    private const int KeyStream = 0, ValueStream = 1;

    private readonly bool _owned;

    private SanaWmWindowedKeys(Tensor keys, Tensor values, bool owned)
    {
        Keys = keys;
        Values = values;
        _owned = owned;
    }

    /// <summary>Head-major keys <c>[1, H, window + chunk, D]</c>.</summary>
    public Tensor Keys { get; }

    /// <summary>Head-major values, shaped like <see cref="Keys"/>.</summary>
    public Tensor Values { get; }

    /// <summary>Joins <paramref name="window"/>'s committed keys and values in front of the chunk's.</summary>
    public static SanaWmWindowedKeys Join(IBackend backend, RollingTokenWindow? window, Tensor keys, Tensor values)
    {
        if (window?.Joined(KeyStream) is not Tensor earlierKeys || window.Joined(ValueStream) is not Tensor earlierValues)
            return new SanaWmWindowedKeys(keys, values, owned: false);
        Tensor joinedKeys = StreamingTensors.Concat(backend, [earlierKeys, keys], TokenAxis);
        try
        {
            return new SanaWmWindowedKeys(joinedKeys, StreamingTensors.Concat(backend, [earlierValues, values], TokenAxis), owned: true);
        }
        catch
        {
            joinedKeys.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!_owned)
            return;
        Keys.Dispose();
        Values.Dispose();
    }
}
