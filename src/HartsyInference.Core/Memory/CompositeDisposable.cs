namespace HartsyInference.Core.Memory;

/// <summary>Disposes several things as one, in the order given, so a caller that owns a set of lifetimes can hand over a single handle.</summary>
/// <remarks><para>A pipeline holds whatever kept its weights alive: a checkpoint's memory map, the widened copies a
/// backend needed, a merged LoRA stack. Each is a separate lifetime with the same owner and the same end, and threading
/// each one through as its own constructor parameter is how those signatures grow a parameter per format.</para>
/// <para>Disposal continues past a failure and rethrows afterwards: a handle that threw must not strand the ones behind
/// it, which on a GPU path is a leaked allocation rather than a leaked object.</para></remarks>
public sealed class CompositeDisposable : IDisposable
{
    private readonly IDisposable?[] _items;
    private int _disposed;

    public CompositeDisposable(params IDisposable?[] items) => _items = items;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        List<Exception>? failures = null;
        foreach (IDisposable? item in _items)
        {
            try
            {
                item?.Dispose();
            }
            catch (Exception error)
            {
                (failures ??= new List<Exception>()).Add(error);
            }
        }
        if (failures is not null)
            throw new AggregateException("One or more handles failed to dispose.", failures);
    }
}
