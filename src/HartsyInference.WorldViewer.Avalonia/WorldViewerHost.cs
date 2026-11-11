using Avalonia;
using Avalonia.Threading;

namespace HartsyInference.WorldViewer.Avalonia;

/// <summary>Opens the live window on its own UI thread, so the caller's generation and spool code run unchanged. Closing the window,
/// or pressing Esc or Q, requests the run's stop the same way <c>q</c> does in the terminal, so the run's output is still saved.</summary>
public sealed class WorldViewerHost : IWorldWindow
{
    private readonly WorldSteering _steering;
    private readonly WorldPlayback _playback;
    private readonly CancellationTokenSource _stop;
    private readonly CancellationTokenSource _exit = new();
    private readonly TaskCompletionSource _ready = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly TaskCompletionSource _closed = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly Thread _thread;
    private readonly object _gate = new();
    private WorldViewerWindow? _window;
    private bool _disposed;

    private WorldViewerHost(WorldSteering steering, WorldPlayback playback, CancellationTokenSource stop)
    {
        ArgumentNullException.ThrowIfNull(steering);
        ArgumentNullException.ThrowIfNull(playback);
        ArgumentNullException.ThrowIfNull(stop);
        _steering = steering;
        _playback = playback;
        _stop = stop;
        _thread = new Thread(Run) { IsBackground = true, Name = "HartsyInference world viewer" };
    }

    /// <summary>Opens the window and returns once it is on screen. Throws <see cref="InvalidOperationException"/> when no display is
    /// reachable or the window cannot start.</summary>
    /// <remarks>Pass the run's stop source. Dispose the host before that source, since closing the window cancels it. Call this once
    /// per process: the first host sets up the process's Avalonia application and its UI thread, and a second host in the same
    /// process is not tested.</remarks>
    public static WorldViewerHost Start(WorldSteering steering, WorldPlayback playback, CancellationTokenSource stop)
    {
        EnsureDisplay();
        WorldViewerHost host = new(steering, playback, stop);
        host._thread.Start();
        try
        {
            host._ready.Task.Wait(TimeSpan.FromSeconds(15));
        }
        catch (AggregateException ex) when (ex.InnerException is not null)
        {
            host.Dispose();
            throw new InvalidOperationException("The world viewer window could not open: " + ex.InnerException.Message, ex.InnerException);
        }
        if (!host._ready.Task.IsCompleted)
        {
            host.Dispose();
            throw new InvalidOperationException("The world viewer window did not open within 15 seconds.");
        }
        return host;
    }

    /// <summary>Completes when the window has closed. A cancelled <paramref name="cancel"/> ends the wait and leaves the window open.</summary>
    public Task WaitUntilClosedAsync(CancellationToken cancel) => _closed.Task.WaitAsync(cancel);

    /// <summary>Closes the window and waits for its thread to finish. Nothing reaches the stop source after this returns.</summary>
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
        }
        if (_thread.IsAlive)
        {
            Dispatcher.UIThread.Post(() => _window?.Close());
        }
        if (!_closed.Task.Wait(TimeSpan.FromSeconds(5)))
        {
            _exit.Cancel();
            _closed.Task.Wait(TimeSpan.FromSeconds(5));
        }
        _thread.Join(TimeSpan.FromSeconds(5));
        _exit.Dispose();
    }

    private void Run()
    {
        try
        {
            AppBuilder builder = AppBuilder.Configure<WorldViewerApplication>().UsePlatformDetect();
            builder.SetupWithoutStarting();
            Dispatcher.UIThread.Post(OpenWindow);
            Dispatcher.UIThread.MainLoop(_exit.Token);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _ready.TrySetException(ex);
            RequestStop();
        }
        finally
        {
            _closed.TrySetResult();
        }
    }

    private void OpenWindow()
    {
        WorldViewerWindow window = new(_steering, _playback, OnWindowClosed);
        _window = window;
        window.Show();
        window.Activate();
        _ready.TrySetResult();
    }

    private void OnWindowClosed()
    {
        _window = null;
        RequestStop();
        _exit.Cancel();
    }

    private void RequestStop()
    {
        lock (_gate)
        {
            if (!_disposed)
            {
                _stop.Cancel();
            }
        }
    }

    private static void EnsureDisplay()
    {
        if (OperatingSystem.IsLinux()
            && string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("DISPLAY"))
            && string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("WAYLAND_DISPLAY")))
        {
            throw new InvalidOperationException(
                "--view needs a display: set DISPLAY (X11) or WAYLAND_DISPLAY. Without one, run without --view to save the frames.");
        }
    }
}
