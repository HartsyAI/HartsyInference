namespace HartsyInference.WorldViewer;

/// <summary>A window that shows one run. The Avalonia host implements it; tests substitute a fake.</summary>
public interface IWorldWindow : IDisposable
{
    /// <summary>Completes when the user closes the window. Cancelling <paramref name="cancel"/> ends the wait and leaves the window open.</summary>
    Task WaitUntilClosedAsync(CancellationToken cancel);
}
