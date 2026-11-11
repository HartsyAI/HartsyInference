using System.Diagnostics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Threading;
using HartsyInference.Engine.Requests;

namespace HartsyInference.WorldViewer.Avalonia;

/// <summary>The live window: frames from <see cref="WorldPlayback"/> on a clock at the stream's frame rate, held keys from the keyboard
/// into <see cref="WorldSteering"/>, and a status block beneath the picture. Esc and Q close it, and closing it requests the run's stop.</summary>
internal sealed class WorldViewerWindow : Window
{
    private readonly WorldSteering _steering;
    private readonly WorldPlayback _playback;
    private readonly Action _requestStop;
    private readonly Image _image = new() { Stretch = Stretch.Uniform };
    private readonly TextBlock _summary = new() { Foreground = Brushes.White, FontFamily = new FontFamily("Consolas,Menlo,Courier New,monospace"), TextWrapping = TextWrapping.Wrap };
    private readonly DispatcherTimer _clock;
    private readonly Stopwatch _elapsed = Stopwatch.StartNew();
    private WriteableBitmap? _bitmap;
    private int _ticks;

    /// <summary>Creates the window over <paramref name="steering"/> and <paramref name="playback"/>; <paramref name="requestStop"/> is
    /// called once when the window closes.</summary>
    public WorldViewerWindow(WorldSteering steering, WorldPlayback playback, Action requestStop)
    {
        ArgumentNullException.ThrowIfNull(steering);
        ArgumentNullException.ThrowIfNull(playback);
        ArgumentNullException.ThrowIfNull(requestStop);
        _steering = steering;
        _playback = playback;
        _requestStop = requestStop;
        _clock = new DispatcherTimer(TimeSpan.FromMilliseconds(8), DispatcherPriority.Render, (_, _) => Tick());
        Title = WorldPlaybackText.Title(steering.Scheme);
        Width = 1024;
        Height = 640;
        Background = Brushes.Black;

        Grid grid = new();
        grid.Children.Add(_image);
        grid.Children.Add(new Border
        {
            VerticalAlignment = VerticalAlignment.Bottom,
            Background = new SolidColorBrush(Color.FromArgb(170, 0, 0, 0)),
            Padding = new Thickness(12),
            Child = _summary,
        });
        Content = grid;

        KeyDown += OnKeyDown;
        KeyUp += OnKeyUp;
        Deactivated += (_, _) => _steering.ReleaseAll();
        Opened += (_, _) => _clock.Start();
        Closed += (_, _) =>
        {
            _clock.Stop();
            _requestStop();
        };
        UpdateSummary();
    }

    private void Tick()
    {
        WorldPaceStep step = _playback.Step(_elapsed.Elapsed, out VideoFrame? frame);
        if (step == WorldPaceStep.Show && frame is not null)
        {
            Draw(frame);
        }
        _ticks++;
        if (_ticks % 25 == 0)
        {
            UpdateSummary();
        }
    }

    private void UpdateSummary() =>
        _summary.Text = WorldPlaybackText.Summary(_playback.Status(), _steering.HeldKeys(), _steering.Scheme);

    private void Draw(VideoFrame frame)
    {
        WriteableBitmap bitmap = _bitmap is { } current && current.PixelSize.Width == frame.Width && current.PixelSize.Height == frame.Height
            ? current
            : CreateBitmap(frame.Width, frame.Height);
        using ILockedFramebuffer target = bitmap.Lock();
        ReadOnlySpan<byte> rgb = frame.Rgb;
        unsafe
        {
            byte* destination = (byte*)target.Address;
            for (int y = 0; y < frame.Height; y++)
            {
                byte* row = destination + (long)y * target.RowBytes;
                int source = y * frame.Width * 3;
                for (int x = 0; x < frame.Width; x++, source += 3)
                {
                    row[x * 4] = rgb[source + 2];
                    row[x * 4 + 1] = rgb[source + 1];
                    row[x * 4 + 2] = rgb[source];
                    row[x * 4 + 3] = 255;
                }
            }
        }
        _image.InvalidateVisual();
    }

    private WriteableBitmap CreateBitmap(int width, int height)
    {
        WriteableBitmap bitmap = new(
            new PixelSize(width, height), new global::Avalonia.Vector(96, 96), PixelFormat.Bgra8888, AlphaFormat.Opaque);
        _bitmap = bitmap;
        _image.Source = bitmap;
        return bitmap;
    }

    private void OnKeyDown(object? sender, KeyEventArgs e)
    {
        e.Handled = true;
        string key = e.Key.ToString();
        if (key is "Escape" or "Q")
        {
            Close();
            return;
        }
        _steering.Press(key);
    }

    private void OnKeyUp(object? sender, KeyEventArgs e)
    {
        e.Handled = true;
        _steering.Release(e.Key.ToString());
    }
}
