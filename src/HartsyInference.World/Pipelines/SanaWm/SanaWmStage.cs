namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>One resident component of a SANA-WM run, held for exactly one stage. Disposing the stage releases the component
/// and everything loaded for it, so the next stage has the device to itself. Release runs once; later calls do nothing.</summary>
/// <typeparam name="T">The component the stage exposes.</typeparam>
public sealed class SanaWmStage<T> : IDisposable
    where T : class
{
    private T? _component;
    private Action? _release;

    /// <summary>Wraps <paramref name="component"/>; <paramref name="release"/> frees it and its weights.</summary>
    public SanaWmStage(T component, Action release)
    {
        ArgumentNullException.ThrowIfNull(component);
        ArgumentNullException.ThrowIfNull(release);
        _component = component;
        _release = release;
    }

    /// <summary>The resident component.</summary>
    /// <exception cref="ObjectDisposedException">The stage was released.</exception>
    public T Component => _component ?? throw new ObjectDisposedException(nameof(SanaWmStage<T>));

    /// <inheritdoc/>
    public void Dispose()
    {
        _component = null;
        Interlocked.Exchange(ref _release, null)?.Invoke();
    }
}
