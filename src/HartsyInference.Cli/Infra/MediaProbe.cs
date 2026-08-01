using System.Diagnostics;
using System.Globalization;

namespace HartsyInference.Cli.Infra;

/// <summary>Reads media dimensions via <c>ffprobe</c> (works for both video containers and still images) —
/// used by <c>hartsy restore --scale</c>, which needs the input size before dispatch to compute the target
/// area. Full decoding still happens once, later, in the dispatch path.</summary>
public static class MediaProbe
{
    /// <summary>Returns (width, height) of the first video/image stream.</summary>
    public static (int Width, int Height) Dimensions(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Input not found: {path}");
        ProcessStartInfo psi = new()
        {
            FileName = "ffprobe",
            Arguments = $"-v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 \"{path}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        using Process proc = Process.Start(psi)
            ?? throw new InvalidOperationException("ffprobe failed to start. Install ffmpeg to use --scale.");
        string output = proc.StandardOutput.ReadToEnd().Trim();
        proc.WaitForExit();
        string[] parts = output.Split(',');
        if (proc.ExitCode != 0 || parts.Length < 2)
            throw new InvalidOperationException($"ffprobe could not read dimensions of '{path}'.");
        return (int.Parse(parts[0], CultureInfo.InvariantCulture), int.Parse(parts[1], CultureInfo.InvariantCulture));
    }
}
