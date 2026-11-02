namespace HartsyInference.Engine;

/// <summary>Holds one cached pipeline for the duration of a request. Disposing it releases the hold exactly once.</summary>
internal sealed class PipelineLease : IDisposable
{
    private readonly string _key;
    private readonly Action<string> _release;
    private int _released;

    public PipelineLease(string key, Action<string> release)
    {
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(release);
        _key = key;
        _release = release;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _released, 1) == 0)
        {
            _release(_key);
        }
    }
}
