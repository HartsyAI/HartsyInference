using HartsyInference.Core.Configuration;

namespace HartsyInference.Video.Encoding;

/// <summary>Finds the <c>ffmpeg</c> and <c>ffprobe</c> binaries the engine launches. Resolution order: the <c>paths.ffmpegDir</c> setting's
/// folder when it holds the tool, then the bare name (the process <c>PATH</c>).</summary>
public static class FfmpegLocator
{
    /// <summary>The command to launch for <paramref name="tool"/> (<c>ffmpeg</c> or <c>ffprobe</c>): the full path inside
    /// <c>paths.ffmpegDir</c> when the tool is there, otherwise <paramref name="tool"/> itself, which the OS resolves on <c>PATH</c>.</summary>
    public static string Resolve(string tool) => Resolve(tool, EngineKnobs.FfmpegDir.Value);

    /// <summary>As <see cref="Resolve(string)"/> with the directory given (null or empty means <c>PATH</c> only).</summary>
    public static string Resolve(string tool, string? directory)
    {
        ArgumentException.ThrowIfNullOrEmpty(tool);
        if (string.IsNullOrWhiteSpace(directory)) return tool;
        string folder = directory.Trim();
        foreach (string name in OperatingSystem.IsWindows() ? new[] { tool + ".exe", tool } : [tool])
        {
            string candidate = Path.Combine(folder, name);
            if (File.Exists(candidate)) return candidate;
        }
        return tool;
    }

    /// <summary>The install hint appended to a launch failure: names the setting that points the engine at a folder.</summary>
    internal const string InstallHint = "Install ffmpeg (for example apt install ffmpeg, or winget install ffmpeg) and put it on PATH, "
        + "or set paths.ffmpegDir to the folder that holds ffmpeg and ffprobe.";
}
