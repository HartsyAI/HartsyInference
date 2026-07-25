using HartsyInference.Core.Logging;

namespace HartsyInference.Cli.Infra;

/// <summary>Discovers models already on disk under the models root, so the CLI can offer a pick-list instead of a typed path.</summary>
/// <remarks>Aware of the SwarmUI/ComfyUI-style folder layout (Stable-Diffusion/, llm/, Music/, clip/, yolo/, …) that installs
/// actually use, not a single flat per-modality folder.</remarks>
public static class LocalModelScanner
{
    // Per modality: which subfolders of the models root to look in, which file extensions count as a single-file model,
    // whether to recurse for those files, and whether an immediate subdirectory counts as a (multi-file) model.
    private sealed record ScanSpec(string[] Roots, string[] FileExts, bool Recurse, bool IncludeDirs);

    private static ScanSpec SpecFor(Modality modality) => modality switch
    {
        Modality.Image => new(new[] { "Stable-Diffusion", "diffusion_models", "unet", "Image" }, new[] { ".safetensors", ".gguf", ".sft" }, Recurse: true, IncludeDirs: false),
        Modality.Text => new(new[] { "llm", "LLM", "Text" }, new[] { ".gguf" }, Recurse: false, IncludeDirs: true),
        Modality.Vision => new(new[] { "clip", "yolo", "Vision" }, new[] { ".safetensors" }, Recurse: false, IncludeDirs: false),
        Modality.Music => new(new[] { "Music", "audiogen", "musicgen" }, new[] { ".safetensors" }, Recurse: false, IncludeDirs: true),
        Modality.Speech => new(new[] { "tts", "TTS", "piper", "Voices", "Audio" }, new[] { ".onnx" }, Recurse: true, IncludeDirs: false),
        Modality.Video => new(new[] { "Video", "video" }, new[] { ".safetensors" }, Recurse: true, IncludeDirs: false),
        Modality.Mesh => new(new[] { "3D", "ThreeD", "threed" }, Array.Empty<string>(), Recurse: false, IncludeDirs: true),
        Modality.World => new(new[] { "Interactive", "World", "world" }, Array.Empty<string>(), Recurse: false, IncludeDirs: true),
        _ => new(Array.Empty<string>(), Array.Empty<string>(), Recurse: false, IncludeDirs: false),
    };

    /// <summary>Lists the models found under the models root for <paramref name="modality"/>, sorted by label and de-duplicated by path.</summary>
    /// <remarks>Empty when nothing local matches.</remarks>
    public static IReadOnlyList<DiscoveredModel> Scan(Modality modality)
    {
        string modelsRoot = RepoPaths.ModelsRoot();
        if (!Directory.Exists(modelsRoot))
            return Array.Empty<DiscoveredModel>();

        ScanSpec spec = SpecFor(modality);
        Dictionary<string, DiscoveredModel> found = new(StringComparer.OrdinalIgnoreCase);

        foreach (string rootName in spec.Roots)
        {
            string root = Path.Combine(modelsRoot, rootName);
            if (!Directory.Exists(root))
                continue;

            if (spec.IncludeDirs)
            {
                foreach (string dir in SafeEnumerateDirectories(root))
                {
                    string name = Path.GetFileName(dir);
                    if (name.StartsWith('.'))
                        continue;
                    Add(found, dir, modelsRoot, isDir: true, 0);
                }
            }

            if (spec.FileExts.Length > 0)
            {
                foreach (string file in EnumerateModelFiles(root, spec.FileExts, spec.Recurse))
                    Add(found, file, modelsRoot, isDir: false, FileSizeOrZero(file));
            }
        }

        List<DiscoveredModel> models = found.Values.ToList();
        models.Sort((a, b) => string.Compare(a.Label, b.Label, StringComparison.OrdinalIgnoreCase));
        return models;
    }

    /// <summary>The folders (relative to the models root) searched for <paramref name="modality"/>, for empty-state hints.</summary>
    public static IReadOnlyList<string> SearchedFolders(Modality modality) => SpecFor(modality).Roots;

    private static void Add(Dictionary<string, DiscoveredModel> into, string path, string modelsRoot, bool isDir, long size)
    {
        string full = Path.GetFullPath(path);
        if (into.ContainsKey(full))
            return;
        string label = Path.GetRelativePath(modelsRoot, full).Replace('\\', '/');
        into[full] = new DiscoveredModel { Label = label, Path = full, IsDirectory = isDir, SizeBytes = size };
    }

    private static IEnumerable<string> EnumerateModelFiles(string root, string[] exts, bool recurse)
    {
        HashSet<string> extSet = new(exts, StringComparer.OrdinalIgnoreCase);
        if (!recurse)
        {
            foreach (string file in SafeEnumerateFiles(root))
                if (extSet.Contains(Path.GetExtension(file)))
                    yield return file;
            yield break;
        }

        // Bounded recursive walk that skips hidden/cache dirs, so a stray huge tree can't stall discovery.
        Stack<(string Dir, int Depth)> stack = new();
        stack.Push((root, 0));
        while (stack.Count > 0)
        {
            (string dir, int depth) = stack.Pop();
            foreach (string file in SafeEnumerateFiles(dir))
                if (extSet.Contains(Path.GetExtension(file)))
                    yield return file;
            if (depth >= 4)
                continue;
            foreach (string sub in SafeEnumerateDirectories(dir))
            {
                string name = Path.GetFileName(sub);
                if (name.StartsWith('.'))
                    continue;
                stack.Push((sub, depth + 1));
            }
        }
    }

    private static IEnumerable<string> SafeEnumerateFiles(string dir)
    {
        try
        {
            return Directory.EnumerateFiles(dir);
        }
        catch (Exception ex)
        {
            Logs.Warning($"Model scan skipped '{dir}': {ex.Message}");
            return Array.Empty<string>();
        }
    }

    private static IEnumerable<string> SafeEnumerateDirectories(string dir)
    {
        try
        {
            return Directory.EnumerateDirectories(dir);
        }
        catch (Exception ex)
        {
            Logs.Warning($"Model scan skipped '{dir}': {ex.Message}");
            return Array.Empty<string>();
        }
    }

    private static long FileSizeOrZero(string file)
    {
        try
        {
            return new FileInfo(file).Length;
        }
        catch (Exception ex)
        {
            Logs.Warning($"Model scan skipped size lookup for '{file}': {ex.Message}");
            return 0;
        }
    }
}
