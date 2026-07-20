using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Features;

/// <summary>Engine-native replacement for the host app's model-set lookups: finds a side-model file by name under the
/// conventional folders of <see cref="RepoPaths.ModelsRoot"/>. Absolute paths that exist are returned untouched, so a
/// caller may always pass either a bare model id or a concrete path.</summary>
public static class ModelFileLocator
{
    /// <summary>Extensions tried, in order, when the supplied name carries none.</summary>
    private static readonly string[] _extensions = [".safetensors", ".sft", ".bin", ".pth", ".pt", ".ckpt"];

    /// <summary>Resolves <paramref name="nameOrPath"/> to an existing file, searching the given models-root-relative
    /// <paramref name="subfolders"/> (recursively) and trying the common weight extensions. Returns null when nothing matches.</summary>
    public static string? Find(string? nameOrPath, params string[] subfolders)
    {
        if (string.IsNullOrWhiteSpace(nameOrPath))
        {
            return null;
        }
        if (File.Exists(nameOrPath))
        {
            return Path.GetFullPath(nameOrPath);
        }
        string root = RepoPaths.ModelsRoot();
        string rooted = Path.Combine(root, nameOrPath);
        if (File.Exists(rooted))
        {
            return rooted;
        }
        foreach (string candidate in NameCandidates(nameOrPath))
        {
            string direct = Path.Combine(root, candidate);
            if (File.Exists(direct))
            {
                return direct;
            }
            foreach (string sub in subfolders)
            {
                string path = Path.Combine(root, sub, candidate);
                if (File.Exists(path))
                {
                    return path;
                }
            }
        }
        // Last resort: a recursive scan of each subfolder matching on the bare stem (the host may nest by author/family).
        string stem = Path.GetFileNameWithoutExtension(nameOrPath);
        foreach (string sub in subfolders)
        {
            string dir = Path.Combine(root, sub);
            if (!Directory.Exists(dir))
            {
                continue;
            }
            try
            {
                foreach (string file in Directory.EnumerateFiles(dir, "*", SearchOption.AllDirectories))
                {
                    if (Path.GetFileNameWithoutExtension(file).Equals(stem, StringComparison.OrdinalIgnoreCase))
                    {
                        return file;
                    }
                }
            }
            catch (Exception ex)
            {
                Logs.Warning($"[Features] Model scan of '{dir}' failed: {ex.GetType().Name}: {ex.Message}.");
            }
        }
        return null;
    }

    /// <summary>Resolves like <see cref="Find"/> but throws a descriptive error instead of returning null.</summary>
    public static string Require(string? nameOrPath, string role, params string[] subfolders)
    {
        return Find(nameOrPath, subfolders)
            ?? throw new InvalidOperationException(
                $"{role} '{nameOrPath}' was not found under '{RepoPaths.ModelsRoot()}' (searched: {string.Join(", ", subfolders)}).");
    }

    private static IEnumerable<string> NameCandidates(string name)
    {
        string ext = Path.GetExtension(name);
        if (!string.IsNullOrEmpty(ext) && Array.Exists(_extensions, e => e.Equals(ext, StringComparison.OrdinalIgnoreCase)))
        {
            yield return name;
            yield break;
        }
        foreach (string e in _extensions)
        {
            yield return name + e;
        }
        yield return name;
    }
}
