namespace HartsyInference.Engine;

/// <summary>Resolves where a generated artifact belongs and writes it there, so every consumer (CLI, HTTP API,
/// extension) lands in the same root with the same auto-numbered naming instead of each inventing its own.</summary>
public static class OutputWriter
{
    /// <summary>The caller's own directory when it names one, else the shared output root.</summary>
    public static string ResolveDir(string? outputDir) =>
        string.IsNullOrWhiteSpace(outputDir) ? RepoPaths.OutputRoot() : Path.GetFullPath(outputDir);

    /// <summary>Writes <paramref name="bytes"/> as <c>&lt;slug&gt;-0001.&lt;ext&gt;</c>, stepping the number past any
    /// existing file. Returns the written path.</summary>
    public static string WriteBytes(byte[] bytes, string? outputDir, string slugSource, string extension)
    {
        string dir = ResolveDir(outputDir);
        Directory.CreateDirectory(dir);
        string slug = Slug.Make(slugSource);
        string ext = extension.TrimStart('.');
        string path = NextAvailablePath(dir, i => $"{slug}-{i:D4}.{ext}", File.Exists, () => $"{slug}-{Guid.NewGuid():N}.{ext}");
        File.WriteAllBytes(path, bytes);
        return path;
    }

    /// <summary>Writes each entry as <c>&lt;name&gt;.&lt;ext&gt;</c> inside one fresh <c>&lt;slug&gt;-0001</c> directory,
    /// so a multi-file result (stems) stays grouped. Returns the created directory.</summary>
    public static string WriteGroup(IReadOnlyDictionary<string, byte[]> files, string? outputDir, string slugSource, string extension)
    {
        string baseDir = ResolveDir(outputDir);
        Directory.CreateDirectory(baseDir);
        string slug = Slug.Make(slugSource);
        string dir = NextAvailablePath(baseDir, i => $"{slug}-{i:D4}", Directory.Exists, () => $"{slug}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        string ext = extension.TrimStart('.');
        foreach (KeyValuePair<string, byte[]> file in files)
            File.WriteAllBytes(Path.Combine(dir, $"{Slug.Make(file.Key)}.{ext}"), file.Value);
        return dir;
    }

    /// <summary>Finds the next non-colliding name by trying <paramref name="nameFor"/>(1), (2), … .</summary>
    /// <remarks>Falls back to a GUID-suffixed name after 100000 tries.</remarks>
    public static string NextAvailablePath(string dir, Func<int, string> nameFor, Func<string, bool> exists, Func<string> fallbackName)
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
