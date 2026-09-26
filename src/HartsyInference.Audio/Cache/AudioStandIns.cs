using System.Runtime.InteropServices;
using System.Text.Json;
using HartsyInference.Core.Logging;

namespace HartsyInference.Audio.Cache;

/// <summary>Lets a converted checkpoint stand in for the upstream files it was made from. A Hartsy artifact names,
/// in its <c>hartsy.stands_in_for</c> metadata, the paths under the audio models root that the engine's loaders
/// open (<c>tts/nari-labs--Dia-1.6B-0626/pytorch_model.bin</c>, <c>music/acestep/acestep-v15-turbo.safetensors</c>);
/// <see cref="Sync"/> links the artifact into each of those paths that is missing, so every loader finds it where it
/// already looks, whatever it is called on disk and whichever of the engine's acquisition paths the family uses.
/// <para>Links are hard links (no copy, and a size check sees the real file); a symbolic link is the fallback when the
/// target is on another filesystem. Links are recorded in <c>.hartsy-standins.json</c> at the root, so removing the
/// artifact removes its links on the next sync instead of leaving them to be mistaken for an upstream download. An
/// existing file is never replaced: a real upstream copy wins over a stand-in.</para></summary>
public static class AudioStandIns
{
    /// <summary>Metadata key listing the audio-root-relative paths an artifact stands in for, separated by <c>;</c>.</summary>
    public const string MetadataKey = "hartsy.stands_in_for";

    private const string ManifestName = ".hartsy-standins.json";
    private static readonly TimeSpan ResyncInterval = TimeSpan.FromSeconds(2);
    private static readonly object _lock = new();
    private static readonly Dictionary<string, DateTime> _lastSync = new(StringComparer.Ordinal);
    private static readonly Dictionary<(string Path, long Length, DateTime Written), string[]> _declared = [];

    /// <summary>Links every artifact under <paramref name="root"/> into its declared paths, once per process; later
    /// calls return immediately. Loaders that open a fixed path call this before looking.</summary>
    public static void EnsureSynced(string root)
    {
        lock (_lock)
        {
            if (_lastSync.ContainsKey(Path.GetFullPath(root)))
            {
                return;
            }
        }
        Sync(root);
    }

    /// <summary>Rescans <paramref name="root"/> unless it was scanned in the last two seconds: for a lookup that just
    /// missed, since an artifact may have been installed while the engine was running.</summary>
    public static void Resync(string root)
    {
        lock (_lock)
        {
            if (_lastSync.TryGetValue(Path.GetFullPath(root), out DateTime last) && DateTime.UtcNow - last < ResyncInterval)
            {
                return;
            }
        }
        Sync(root);
    }

    /// <summary>Removes links whose artifact is gone, then links each artifact into its declared paths that do not
    /// exist. Returns the number of links created. Never throws: a failure is logged and the loader's own
    /// not-found handling takes over.</summary>
    public static int Sync(string root)
    {
        string fullRoot = Path.GetFullPath(root);
        lock (_lock)
        {
            _lastSync[fullRoot] = DateTime.UtcNow;
            try
            {
                return SyncLocked(fullRoot);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException)
            {
                Logs.Warning($"[AudioCache] Could not link converted checkpoints under '{fullRoot}': {ex.Message}");
                return 0;
            }
        }
    }

    /// <summary>The paths <paramref name="safetensorsPath"/> declares it stands in for, or empty. Paths that are rooted
    /// or climb out of the root are dropped: a file must not be able to write outside the models tree.</summary>
    public static IReadOnlyList<string> ReadDeclared(string safetensorsPath)
    {
        string? value = ReadMetadataValue(safetensorsPath, MetadataKey);
        if (string.IsNullOrWhiteSpace(value))
        {
            return [];
        }
        List<string> paths = [];
        foreach (string raw in value.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            string relative = raw.Replace('\\', '/');
            if (Path.IsPathRooted(relative) || relative.Split('/').Any(s => s is ".." or "." or ""))
            {
                Logs.Warning($"[AudioCache] '{Path.GetFileName(safetensorsPath)}' declares '{raw}', which is not a path inside the models root; ignored.");
                continue;
            }
            paths.Add(relative);
        }
        return paths;
    }

    private static int SyncLocked(string root)
    {
        if (!Directory.Exists(root))
        {
            return 0;
        }
        string manifestPath = Path.Combine(root, ManifestName);
        Dictionary<string, string> links = ReadManifest(manifestPath);
        bool changed = false;
        // A link whose artifact is gone would otherwise pass for an upstream download and keep declaring itself.
        foreach ((string link, string target) in links.ToList())
        {
            string linkPath = Path.Combine(root, link);
            string targetPath = Path.Combine(root, target);
            if (File.Exists(targetPath) && File.Exists(linkPath))
            {
                continue;
            }
            if (!File.Exists(targetPath))
            {
                TryDelete(linkPath);
            }
            links.Remove(link);
            changed = true;
        }
        HashSet<string> linkSet = new(links.Keys.Select(l => Path.Combine(root, l)), StringComparer.Ordinal);
        int created = 0;
        foreach (string file in Directory.EnumerateFiles(root, "*.safetensors", new EnumerationOptions { RecurseSubdirectories = true, IgnoreInaccessible = true }))
        {
            if (linkSet.Contains(file))
            {
                continue;
            }
            foreach (string relative in Declared(file))
            {
                string destination = Path.Combine(root, relative);
                if (File.Exists(destination) || Directory.Exists(destination))
                {
                    continue;
                }
                Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
                bool hard = TryHardLink(file, destination);
                if (!hard)
                {
                    File.CreateSymbolicLink(destination, file);
                }
                links[relative] = Path.GetRelativePath(root, file).Replace('\\', '/');
                linkSet.Add(destination);
                changed = true;
                created++;
                Logs.Info($"[AudioCache] {relative} -> {links[relative]} ({(hard ? "hard link" : "symlink")}).");
            }
        }
        if (changed)
        {
            WriteManifest(manifestPath, links);
        }
        return created;
    }

    private static string[] Declared(string file)
    {
        FileInfo info = new(file);
        (string, long, DateTime) key = (file, info.Length, info.LastWriteTimeUtc);
        if (!_declared.TryGetValue(key, out string[]? paths))
        {
            try
            {
                paths = [.. ReadDeclared(file)];
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException or InvalidDataException)
            {
                Logs.Debug($"[AudioCache] Skipping unreadable '{file}': {ex.Message}");
                paths = [];
            }
            _declared[key] = paths;
        }
        return paths;
    }

    /// <summary>Reads one <c>__metadata__</c> value from a safetensors header without touching the tensors.</summary>
    private static string? ReadMetadataValue(string path, string key)
    {
        using FileStream stream = new(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        Span<byte> head = stackalloc byte[8];
        if (stream.Read(head) < 8)
        {
            return null;
        }
        long length = BitConverter.ToInt64(head);
        if (length <= 1 || length > Math.Min(stream.Length - 8, 100L << 20))
        {
            throw new InvalidDataException("not a safetensors header");
        }
        byte[] header = new byte[length];
        stream.ReadExactly(header);
        using JsonDocument doc = JsonDocument.Parse(header);
        return doc.RootElement.TryGetProperty("__metadata__", out JsonElement meta) && meta.ValueKind == JsonValueKind.Object
            && meta.TryGetProperty(key, out JsonElement value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
    }

    private static Dictionary<string, string> ReadManifest(string path)
    {
        if (!File.Exists(path))
        {
            return new(StringComparer.Ordinal);
        }
        Dictionary<string, string>? links = JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllBytes(path));
        return links is null ? new(StringComparer.Ordinal) : new(links, StringComparer.Ordinal);
    }

    private static void WriteManifest(string path, Dictionary<string, string> links)
    {
        string temp = path + ".tmp";
        File.WriteAllText(temp, JsonSerializer.Serialize(new SortedDictionary<string, string>(links, StringComparer.Ordinal),
            new JsonSerializerOptions { WriteIndented = true }));
        File.Move(temp, path, overwrite: true);
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (File.Exists(path) || new FileInfo(path).LinkTarget is not null)
            {
                File.Delete(path);
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Logs.Debug($"[AudioCache] Could not remove stale link '{path}': {ex.Message}");
        }
    }

    private static bool TryHardLink(string existing, string link)
    {
        try
        {
            return OperatingSystem.IsWindows() ? CreateHardLinkW(link, existing, IntPtr.Zero) : LinkUnix(existing, link) == 0;
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
        {
            return false;
        }
    }

    [DllImport("libc", EntryPoint = "link", SetLastError = true)]
    private static extern int LinkUnix([MarshalAs(UnmanagedType.LPUTF8Str)] string existing, [MarshalAs(UnmanagedType.LPUTF8Str)] string link);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool CreateHardLinkW(string link, string existing, IntPtr securityAttributes);
}
