using System.Globalization;
using HartsyInference.Engine.Worlds;

namespace HartsyInference.WorldViewer;

/// <summary>The viewer's status text, built from a playback snapshot and the held keys so every front end words it the same way.</summary>
public static class WorldPlaybackText
{
    /// <summary>The window title for <paramref name="scheme"/>, or the bare product name when there is none.</summary>
    public static string Title(IWorldControlScheme? scheme) =>
        scheme is null ? "HartsyInference world viewer" : $"HartsyInference world viewer · {scheme.Name}";

    /// <summary>The status block: phase, buffer, delivery timing, held keys, the scheme's note and its legend.</summary>
    public static string Summary(WorldPlaybackStatus status, IReadOnlyList<string> heldKeys, IWorldControlScheme? scheme)
    {
        ArgumentNullException.ThrowIfNull(heldKeys);
        List<string> lines = new(6);
        lines.Add(PhaseLine(status));
        lines.Add(
            $"Buffered {status.BufferedSeconds.ToString("0.0", CultureInfo.InvariantCulture)} s "
            + $"({status.BufferedFrames}/{status.BufferCapacity} frames) · shown {status.FramesShown} · generated {status.FramesReceived}");
        lines.Add(
            $"Delivered {status.DeliveredPerSecond.ToString("0.0", CultureInfo.InvariantCulture)} fps against "
            + $"{status.Fps} to play");
        if (scheme is null)
        {
            lines.Add("No live control for this model: the clip plays when the run completes.");
            return string.Join('\n', lines);
        }
        lines.Add("Held: " + (heldKeys.Count == 0 ? "none" : string.Join(", ", heldKeys)));
        lines.Add(scheme.Note);
        lines.Add(Legend(scheme.Controls));
        return string.Join('\n', lines);
    }

    private static string PhaseLine(WorldPlaybackStatus status) => status.Phase switch
    {
        WorldPlaybackPhase.Generating => "Generating the first chunk…",
        WorldPlaybackPhase.Playing => "Playing",
        WorldPlaybackPhase.Rebuffering => "Waiting for generation: playback ran out of frames",
        WorldPlaybackPhase.Finished => "Done: the run finished. Press Esc to close.",
        WorldPlaybackPhase.Failed => "Failed: " + (status.Error ?? "the stream stopped"),
        _ => string.Empty,
    };

    private static string Legend(IReadOnlyList<WorldControl> controls)
    {
        List<string> parts = new(controls.Count);
        foreach (WorldControl control in controls)
        {
            parts.Add(control.Key + " " + control.Label);
        }
        return string.Join("  ·  ", parts);
    }
}
