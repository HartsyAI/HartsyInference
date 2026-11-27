using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Streaming;

/// <summary>One tensor of carried state: a recurrent state, or any value a chunk hands to the next. The slot owns its
/// tensors and pins them on the backend, so a device copy survives an activation sweep between chunks.</summary>
public sealed class StreamingTensorSlot : IStreamingState
{
    private readonly IBackend _backend;
    private Tensor? _current;
    private Tensor? _staged;
    private int _disposed;

    /// <summary>Creates an empty slot on <paramref name="backend"/>.</summary>
    public StreamingTensorSlot(IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        _backend = backend;
    }

    /// <summary>The committed tensor, or null before the first commit. Borrowed: valid until the next commit or reset.</summary>
    public Tensor? Current => _current;

    /// <summary>Whether a write is staged.</summary>
    public bool HasStaged => _staged is not null;

    /// <summary>Stages <paramref name="next"/> as the value after the current chunk. The slot takes ownership; an earlier
    /// staged value is released.</summary>
    public void Stage(Tensor next)
    {
        ArgumentNullException.ThrowIfNull(next);
        if (Volatile.Read(ref _disposed) != 0)
        {
            next.Dispose();
            throw new ObjectDisposedException(nameof(StreamingTensorSlot));
        }

        Release(ref _staged);
        _backend.PinActivation(next);
        _staged = next;
    }

    /// <inheritdoc />
    public void Commit()
    {
        if (_staged is null)
            return;
        Release(ref _current);
        _current = _staged;
        _staged = null;
    }

    /// <inheritdoc />
    public void Discard() => Release(ref _staged);

    /// <inheritdoc />
    public void Reset()
    {
        Release(ref _staged);
        Release(ref _current);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Reset();
    }

    private void Release(ref Tensor? tensor)
    {
        if (tensor is null)
            return;
        _backend.UnpinActivation(tensor);
        tensor.Dispose();
        tensor = null;
    }
}
