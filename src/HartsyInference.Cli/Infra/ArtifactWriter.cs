using System.Text;
using HartsyInference.Cli.Dispatch;

namespace HartsyInference.Cli.Infra;

/// <summary>Persists a <see cref="GeneratedArtifact"/> to disk with an auto-numbered, prompt-derived filename.</summary>
public static class ArtifactWriter
{
    /// <summary>Writes <paramref name="artifact"/> under <paramref name="outputDir"/> (or the default output root).</summary>
    /// <remarks>Text with no file bytes is only written when <paramref name="force"/> is set.</remarks>
    /// <returns>The written path, or null when nothing was written.</returns>
    public static string? Write(GeneratedArtifact artifact, string? outputDir, string promptSlug, bool force)
    {
        if (artifact.SelfWritten)
            return null;   // handler already wrote its own file(s) (frames/stems) — nothing more to do.

        bool hasBytes = artifact.FileBytes is { Length: > 0 };
        if (!hasBytes && !force)
            return null;

        string dir = string.IsNullOrWhiteSpace(outputDir) ? RepoPaths.OutputRoot() : Path.GetFullPath(outputDir);
        Directory.CreateDirectory(dir);
        string slug = Slug.Make(promptSlug);
        string ext = artifact.Extension.TrimStart('.');
        string path = NextAvailablePath(dir, i => $"{slug}-{i:D4}.{ext}", File.Exists, () => $"{slug}-{Guid.NewGuid():N}.{ext}");

        if (hasBytes)
            File.WriteAllBytes(path, artifact.FileBytes!);
        else
            File.WriteAllText(path, artifact.Text ?? "", new UTF8Encoding(false));

        return path;
    }

    /// <summary>Finds the next non-colliding path under <paramref name="dir"/> by trying <paramref name="nameFor"/>(1), (2), … .</summary>
    /// <remarks>Falls back to a GUID-suffixed name after 100000 tries.</remarks>
    internal static string NextAvailablePath(string dir, Func<int, string> nameFor, Func<string, bool> exists, Func<string> fallbackName)
    {
        for (int i = 1; i < 100000; i++)
        {
            string candidate = Path.Combine(dir, nameFor(i));
            if (!exists(candidate))
                return candidate;
        }
        return Path.Combine(dir, fallbackName());
    }
}
