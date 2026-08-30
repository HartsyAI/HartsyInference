using System.Collections.ObjectModel;

namespace HartsyInference.Engine.Planning;

/// <summary>Binds construction artifacts to the file state inspected by a video plan.</summary>
internal static class VideoArtifactFileBinding
{
    /// <summary>Captures every resolved file component without retaining an open handle.</summary>
    internal static IReadOnlyDictionary<string, VideoArtifactFileStamp> Capture(
        IReadOnlyDictionary<string, string> componentPaths)
    {
        ArgumentNullException.ThrowIfNull(componentPaths);
        Dictionary<string, VideoArtifactFileStamp> stamps = new Dictionary<string, VideoArtifactFileStamp>(
            componentPaths.Count, StringComparer.Ordinal);
        foreach ((string role, string path) in componentPaths)
        {
            if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            {
                continue;
            }
            stamps.Add(role, CaptureFile(path));
        }
        return new ReadOnlyDictionary<string, VideoArtifactFileStamp>(stamps);
    }

    /// <summary>Rejects a plan when any construction file was replaced or modified after inspection.</summary>
    internal static void RequireUnchanged(VideoPlan plan)
    {
        ArgumentNullException.ThrowIfNull(plan);
        foreach ((string role, VideoArtifactFileStamp expected) in plan.ArtifactFileStamps)
        {
            if (!plan.ComponentPaths.TryGetValue(role, out string? requestedPath)
                || string.IsNullOrWhiteSpace(requestedPath) || !File.Exists(requestedPath))
            {
                throw Changed(role, expected.CanonicalPath);
            }

            VideoArtifactFileStamp current = CaptureFile(requestedPath);
            StringComparison comparison = OperatingSystem.IsWindows()
                ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
            if (!string.Equals(current.CanonicalPath, expected.CanonicalPath, comparison)
                || current.Length != expected.Length || current.LastWriteUtcTicks != expected.LastWriteUtcTicks)
            {
                throw Changed(role, expected.CanonicalPath);
            }
        }
    }

    private static VideoArtifactFileStamp CaptureFile(string path)
    {
        string canonical = VideoCheckpointHashCache.CanonicalPath(path);
        FileInfo info = new FileInfo(canonical);
        if (!info.Exists)
        {
            throw new FileNotFoundException("Planned video artifact was not found.", canonical);
        }
        return new VideoArtifactFileStamp(canonical, info.Length, info.LastWriteTimeUtc.Ticks);
    }

    private static InvalidOperationException Changed(string role, string path) => new(
        $"Video artifact '{role}' changed after planning ('{path}'). Re-plan before generation so profile, "
        + "cache identity, formats, and execution metadata match the bytes that will be loaded.");
}
