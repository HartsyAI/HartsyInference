using HartsyInference.Core.IO;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Features;

/// <summary>Engine-native replacement for the host app's model-set lookups: finds a side-model file by name under the conventional folders of <see cref="RepoPaths.ModelsRoot"/>. Absolute paths that exist are returned untouched, so a caller may always pass either a bare model id or a concrete path.</summary>
public static class ModelFileLocator
{
    /// <summary>Extensions tried, in order, when the supplied name carries none.</summary>
    private static readonly string[] _extensions = [".safetensors", ".sft", ".bin", ".pth", ".pt", ".ckpt"];

    /// <summary>Resolves <paramref name="nameOrPath"/> to an existing file, searching the given models-root-relative <paramref name="subfolders"/> (recursively) and trying the common weight extensions; a subfolder that is missing under its own spelling but exists in another case is searched only after the spelled ones found nothing. Returns null when nothing matches.</summary>
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
        string[] folders = Array.ConvertAll(subfolders, sub => Path.Combine(root, sub));
        // Case variants come second, so a file the spelled folders hold is still the one returned.
        return Search(nameOrPath, root, folders) ?? Search(nameOrPath, null, CaseVariants(root, subfolders, folders));
    }

    /// <summary>Resolves like <see cref="Find"/> but throws a descriptive error instead of returning null.</summary>
    public static string Require(string? nameOrPath, string role, params string[] subfolders)
    {
        return Find(nameOrPath, subfolders)
            ?? throw new InvalidOperationException(
                $"{role} '{nameOrPath}' was not found under '{RepoPaths.ModelsRoot()}' (searched: {string.Join(", ", subfolders)}).");
    }

    /// <summary>Each name candidate directly under <paramref name="root"/> (when given) and in each folder, then a
    /// recursive scan of the folders for the bare stem.</summary>
    private static string? Search(string nameOrPath, string? root, string[] folders)
    {
        foreach (string candidate in NameCandidates(nameOrPath))
        {
            if (root is not null)
            {
                string direct = Path.Combine(root, candidate);
                if (File.Exists(direct))
                {
                    return direct;
                }
            }
            foreach (string folder in folders)
            {
                string path = Path.Combine(folder, candidate);
                if (File.Exists(path))
                {
                    return path;
                }
            }
        }
        // Last resort: a recursive scan of each subfolder matching on the bare stem (the host may nest by author/family).
        string stem = Path.GetFileNameWithoutExtension(nameOrPath);
        foreach (string dir in folders)
        {
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

    /// <summary>The subfolders missing under their own spelling that exist in another case, resolved; always empty on
    /// a case-insensitive filesystem.</summary>
    private static string[] CaseVariants(string root, string[] subfolders, string[] spelled)
    {
        List<string>? variants = null;
        for (int i = 0; i < subfolders.Length; i++)
        {
            string resolved = CaseInsensitivePath.ResolveDirectory(root, subfolders[i]);
            if (!string.Equals(resolved, spelled[i], StringComparison.Ordinal) && Directory.Exists(resolved))
            {
                (variants ??= []).Add(resolved);
            }
        }
        return variants is null ? [] : [.. variants];
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
