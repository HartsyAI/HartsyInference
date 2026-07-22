using System.Text;
using HartsyInference.Cli.Dispatch;

namespace HartsyInference.Cli.Infra;

/// <summary>Persists a <see cref="GeneratedArtifact"/> to disk with an auto-numbered, prompt-derived filename.</summary>
public static class ArtifactWriter
{
    /// <summary>Writes <paramref name="artifact"/> under <paramref name="outputDir"/> (or the default output root).
    /// Text with no file bytes is only written when <paramref name="force"/> is set. Returns the path, or null when
    /// nothing was written.</summary>
    public static string? Write(GeneratedArtifact artifact, string? outputDir, string promptSlug, bool force)
    {
        if (artifact.SelfWritten)
            return null;   // handler already wrote its own file(s) (frames/stems) — nothing more to do.

        bool hasBytes = artifact.FileBytes is { Length: > 0 };
        if (!hasBytes && !force)
            return null;

        string dir = string.IsNullOrWhiteSpace(outputDir) ? RepoPaths.OutputRoot() : Path.GetFullPath(outputDir);
        Directory.CreateDirectory(dir);
        string name = NextName(dir, Slug.Make(promptSlug), artifact.Extension);
        string path = Path.Combine(dir, name);

        if (hasBytes)
            File.WriteAllBytes(path, artifact.FileBytes!);
        else
            File.WriteAllText(path, artifact.Text ?? "", new UTF8Encoding(false));

        return path;
    }

    private static string NextName(string dir, string slug, string extension)
    {
        string ext = extension.TrimStart('.');
        for (int i = 1; i < 100000; i++)
        {
            string candidate = $"{slug}-{i:D4}.{ext}";
            if (!File.Exists(Path.Combine(dir, candidate)))
                return candidate;
        }
        return $"{slug}-{Guid.NewGuid():N}.{ext}";
    }
}
