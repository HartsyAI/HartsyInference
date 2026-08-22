using HartsyInference.Core.Logging;
using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Vision;

/// <summary>Locates detector/segmenter weights under the engine models root using the same folder conventions the SwarmUI extension used (<c>rtdetr</c>, <c>grounding-dino</c>, <c>clipseg</c>, <c>yolov8</c>, <c>sam2</c>). An explicit path supplied by the caller (<c>ModelSpec.LocalPath</c>) always wins; nothing is ever downloaded.</summary>
public static class VisionModelPaths
{
    /// <summary>Models-root subfolder holding RT-DETR checkpoints.</summary>
    public const string RtDetrFolder = "rtdetr";

    /// <summary>Models-root subfolder holding the Grounding DINO snapshot.</summary>
    public const string GroundingDinoFolder = "grounding-dino";

    /// <summary>Models-root subfolder holding the CLIPSeg snapshot.</summary>
    public const string ClipSegFolder = "clipseg";

    /// <summary>Models-root subfolder holding YOLO safetensors.</summary>
    public const string YoloFolder = "yolov8";

    /// <summary>Models-root subfolder holding SAM 2 checkpoints.</summary>
    public const string Sam2Folder = "sam2";

    /// <summary>Finds a single <c>.safetensors</c> for a detector: an explicit file wins, an explicit directory is scanned, otherwise the conventional models-root subfolder (and one nested level) is scanned. Null when absent.</summary>
    public static string? FindCheckpoint(string? explicitPath, string subfolder)
    {
        if (!string.IsNullOrWhiteSpace(explicitPath))
        {
            if (File.Exists(explicitPath))
            {
                return Path.GetFullPath(explicitPath);
            }
            if (Directory.Exists(explicitPath))
            {
                string? found = FirstSafetensors(explicitPath);
                if (found is not null)
                {
                    return found;
                }
            }
        }
        return FirstSafetensors(Path.Combine(RepoPaths.ModelsRoot(), subfolder));
    }

    /// <summary>Finds the Grounding DINO <c>model.safetensors</c> + BERT <c>vocab.txt</c> pair; both must sit in the same directory (the HF snapshot layout, optionally nested one level). Returns nulls when not installed.</summary>
    public static (string? Checkpoint, string? Vocab) FindGroundingDino(string? explicitPath)
    {
        foreach (string dir in CandidateDirectories(explicitPath, GroundingDinoFolder))
        {
            string ckpt = Path.Combine(dir, "model.safetensors");
            string vocab = Path.Combine(dir, "vocab.txt");
            if (File.Exists(ckpt) && File.Exists(vocab))
            {
                return (ckpt, vocab);
            }
        }
        return (null, null);
    }

    /// <summary>Finds the CLIPSeg model directory (the one containing <c>model.safetensors</c>). Null when absent.</summary>
    public static string? FindClipSegDirectory(string? explicitPath)
    {
        foreach (string dir in CandidateDirectories(explicitPath, ClipSegFolder))
        {
            if (File.Exists(Path.Combine(dir, "model.safetensors")))
            {
                return dir;
            }
        }
        return null;
    }

    /// <summary>Finds a YOLO checkpoint by bare model name (extension optional) across the conventional folders, or falls back to the first checkpoint in <paramref name="explicitPath"/>. Null when absent.</summary>
    public static string? FindYolo(string? nameOrPath, string? explicitPath)
    {
        // A checkpoint the caller named explicitly always wins over a name probe — otherwise a stray
        // same-named file in the models folder silently shadows the file the user actually asked for.
        string? explicitResolved = FindCheckpoint(explicitPath, YoloFolder);
        if (explicitResolved is not null)
        {
            return explicitResolved;
        }
        string? byName = ModelFileLocator.Find(nameOrPath, YoloFolder, "yolo", "Yolo");
        return PreferSafetensors(byName);
    }

    /// <summary>Redirects an Ultralytics <c>.pt</c> hit to a converted safetensors sibling when one exists; the engine reads safetensors, so returning the pickle would only fail later with a confusing header error.</summary>
    private static string? PreferSafetensors(string? path)
    {
        if (path is null || !path.EndsWith(".pt", StringComparison.OrdinalIgnoreCase))
        {
            return path;
        }
        string stem = path[..^3];
        foreach (string candidate in new[] { stem + "-folded.safetensors", stem + ".safetensors" })
        {
            if (File.Exists(candidate))
            {
                Logs.Info($"[Vision] '{Path.GetFileName(path)}' is an Ultralytics pickle; using '{Path.GetFileName(candidate)}' instead.");
                return candidate;
            }
        }
        return path;
    }

    /// <summary>Finds a SAM 2 checkpoint used to refine boxes into pixel-accurate masks. Null when absent — callers must fall back to rasterizing the bounding box, SAM 2 is always optional.</summary>
    public static string? FindSam2(string? explicitPath) => FindCheckpoint(explicitPath, Sam2Folder);

    /// <summary>Directories to probe for a snapshot-style model: the explicit path (and its children), then the conventional models-root subfolder (and its children).</summary>
    private static IEnumerable<string> CandidateDirectories(string? explicitPath, string subfolder)
    {
        if (!string.IsNullOrWhiteSpace(explicitPath))
        {
            string? baseDir = Directory.Exists(explicitPath) ? explicitPath : Path.GetDirectoryName(explicitPath);
            if (!string.IsNullOrEmpty(baseDir) && Directory.Exists(baseDir))
            {
                yield return baseDir;
                foreach (string sub in SafeChildren(baseDir))
                {
                    yield return sub;
                }
            }
        }
        string root = Path.Combine(RepoPaths.ModelsRoot(), subfolder);
        if (Directory.Exists(root))
        {
            yield return root;
            foreach (string sub in SafeChildren(root))
            {
                yield return sub;
            }
        }
    }

    private static string? FirstSafetensors(string directory)
    {
        if (!Directory.Exists(directory))
        {
            return null;
        }
        foreach (string dir in new[] { directory }.Concat(SafeChildren(directory)))
        {
            try
            {
                foreach (string file in Directory.EnumerateFiles(dir, "*.safetensors").OrderBy(x => x, StringComparer.Ordinal))
                {
                    return file;
                }
            }
            catch (Exception ex)
            {
                Logs.Warning($"[Vision] Scan of '{dir}' failed: {ex.GetType().Name}: {ex.Message}.");
            }
        }
        return null;
    }

    private static IEnumerable<string> SafeChildren(string directory)
    {
        try
        {
            return Directory.EnumerateDirectories(directory).OrderBy(x => x, StringComparer.Ordinal).ToList();
        }
        catch (Exception ex)
        {
            Logs.Warning($"[Vision] Listing '{directory}' failed: {ex.GetType().Name}: {ex.Message}.");
            return [];
        }
    }
}
