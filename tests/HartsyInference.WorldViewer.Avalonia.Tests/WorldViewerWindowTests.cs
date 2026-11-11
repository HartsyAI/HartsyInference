using Avalonia;
using Avalonia.Controls;
using Avalonia.Headless;
using Avalonia.Headless.XUnit;
using Avalonia.Input;
using System.Runtime.CompilerServices;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Worlds;
using HartsyInference.WorldViewer;
using HartsyInference.WorldViewer.Avalonia;

[assembly: AvaloniaTestApplication(typeof(HartsyInference.WorldViewer.Avalonia.Tests.TestAppBuilder))]
[assembly: CollectionBehavior(DisableTestParallelization = true)]

namespace HartsyInference.WorldViewer.Avalonia.Tests;

public sealed class TestAppBuilder
{
    public static AppBuilder BuildAvaloniaApp() => AppBuilder.Configure<Application>().UseHeadless(new AvaloniaHeadlessPlatformOptions());
}

public sealed class WorldViewerWindowTests
{
    [AvaloniaFact]
    public void Held_keys_steer_the_session_and_releasing_them_stops_it()
    {
        RecordingSession session = new();
        WorldSteering steering = new(session, SanaWmControlScheme.Instance);
        WorldViewerWindow window = new(steering, new WorldPlayback(16, 32), () => { });
        window.Show();

        Down(window, Key.W);
        Up(window, Key.W);

        Assert.Equal(["w", "none"], session.Actions);
        Assert.Empty(steering.HeldKeys());
        window.Close();
    }

    [AvaloniaFact]
    public void Escape_and_closing_request_the_stop_once()
    {
        RecordingSession session = new();
        int stops = 0;
        WorldViewerWindow window = new(new WorldSteering(session, SanaWmControlScheme.Instance), new WorldPlayback(16, 32), () => stops++);
        window.Show();

        Down(window, Key.Escape);
        window.Close();

        Assert.Equal(1, stops);
    }

    [AvaloniaFact]
    public void A_key_the_scheme_does_not_declare_changes_nothing()
    {
        RecordingSession session = new();
        WorldViewerWindow window = new(new WorldSteering(session, SanaWmControlScheme.Instance), new WorldPlayback(16, 32), () => { });
        window.Show();

        Down(window, Key.Space);

        Assert.Empty(session.Actions);
        window.Close();
    }

    private static void Down(Window window, Key key) =>
        window.RaiseEvent(new KeyEventArgs { RoutedEvent = InputElement.KeyDownEvent, Key = key, Source = window });

    private static void Up(Window window, Key key) =>
        window.RaiseEvent(new KeyEventArgs { RoutedEvent = InputElement.KeyUpEvent, Key = key, Source = window });

    private sealed class RecordingSession : IWorldSession
    {
        public List<string> Actions { get; } = [];

        public void SendAction(string action) => Actions.Add(action);

        public async IAsyncEnumerable<VideoFrame> StreamAsync([EnumeratorCancellation] CancellationToken cancel)
        {
            await Task.CompletedTask;
            cancel.ThrowIfCancellationRequested();
            yield break;
        }

        public void Dispose()
        {
        }
    }
}
