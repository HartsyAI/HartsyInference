using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Streaming;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>The keys and values one softmax attention call of a chunk-causal stream reads: the window kept from earlier
/// chunks, then the chunk's own. Without a window, or while it is empty (a stream's first chunk), these are the chunk's own
/// tensors, borrowed; otherwise they are joined copies this scope owns. The window holds its keys and values in BF16; the
/// parts are widened to the chunk's dtype as they are joined, so the attention call sees the same F32 operands whether or not
/// there was a window (the values are exactly the BF16 numbers the call's own narrowing would have produced).</summary>
internal readonly struct SanaWmWindowedKeys : IDisposable
{
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

    /// <summary>Joins <paramref name="window"/>'s committed keys and values in front of the chunk's, one concatenation each.</summary>
    public static SanaWmWindowedKeys Join(IBackend backend, RollingTokenWindow? window, Tensor keys, Tensor values)
    {
        if (window is null || window.IsEmpty)
            return new SanaWmWindowedKeys(keys, values, owned: false);
        Tensor joinedKeys = Joined(backend, window, KeyStream, keys);
        try
        {
            return new SanaWmWindowedKeys(joinedKeys, Joined(backend, window, ValueStream, values), owned: true);
        }
        catch
        {
            joinedKeys.Dispose();
            throw;
        }
    }

    private static Tensor Joined(IBackend backend, RollingTokenWindow window, int stream, Tensor chunk)
    {
        Tensor[] parts = new Tensor[window.PartCount + 1];
        int count = window.CopyParts(stream, parts);
        Tensor?[] widened = new Tensor?[count];
        try
        {
            for (int i = 0; i < count; i++)
            {
                if (parts[i].DType == chunk.DType)
                    continue;
                widened[i] = StreamingTensors.CastTo(backend, parts[i], chunk.DType);
                parts[i] = widened[i]!;
            }

            parts[count] = chunk;
            return StreamingTensors.Concat(backend, parts, window.TokenAxis);
        }
        finally
        {
            foreach (Tensor? copy in widened)
                copy?.Dispose();
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
