using System.Text.Json;
using System.Text.RegularExpressions;
using HartsyInference.Core.Exceptions;
using HartsyInference.ModelAssets.Quant;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>How a text model package is stored on disk.</summary>
public enum TextPackageFormat
{
    /// <summary>A Hugging Face checkpoint directory: a config, a shard index or safetensors shards, and optionally a tokenizer.</summary>
    SafetensorsShards,

    /// <summary>One GGUF file.</summary>
    Gguf,

    /// <summary>A llama.cpp split GGUF: every <c>name-00001-of-000NN.gguf</c> part of one model.</summary>
    SplitGguf,
}

/// <summary>A companion file that goes with a package but is not a model of its own, such as a vision tower (<c>mmproj</c>).</summary>
public sealed record TextPackageSidecar(string Role, string Path);

/// <summary>Whether a package can run DSpark speculation, and the reason when it cannot. The model list (<c>/v1/models</c>) gives the same reason for a format, so a
/// checkpoint with a draft head reads the same on both routes. Without one, this route names the missing head, while the model list cannot see the checkpoint.</summary>
public sealed record TextPackageSpeculation(bool Supported, string Reason)
{
    /// <summary>A GGUF file or split.</summary>
    public const string GgufReason = "GGUF packages carry no DSpark draft head";

    /// <summary>An MLX checkpoint, draft head or not.</summary>
    public const string MlxReason = "MLX checkpoints are not served with DSpark speculation";

    /// <summary>A safetensors checkpoint with a draft head, which serving does not run yet.</summary>
    public const string NotWiredReason = "DSpark speculation is not wired into serving yet";

    /// <summary>A safetensors checkpoint without draft (<c>mtp</c>) tensors.</summary>
    public const string NoDraftHeadReason = "no DSpark draft head (mtp) in this checkpoint";

    /// <summary>A catalog model with no variant at all.</summary>
    public const string NoVariantReason = "no variant ships a DSpark draft head";
}

/// <summary>One text model found on disk. <see cref="EntryPath"/> is the entry point: the checkpoint directory, the GGUF file, or, for a split, its first part.</summary>
public sealed record TextModelPackage(
    string Id,
    string EntryPath,
    TextPackageFormat Format,
    IReadOnlyList<string> Files,
    long TotalBytes,
    string? Family,
    string? Quant,
    IReadOnlyList<TextPackageSidecar> Sidecars,
    TextPackageSpeculation Speculation,
    IReadOnlyList<string> Problems);

/// <summary>What one scan found: the packages, ordered by <see cref="TextModelPackage.Id"/>, and each directory it skipped because it could not be read, with the reason.</summary>
public sealed record TextPackageScan(IReadOnlyList<TextModelPackage> Packages, IReadOnlyList<string> Problems);

/// <summary>Finds the text model packages under a directory tree, from file names, checkpoint configs and shard indexes; no weight data is read.</summary>
/// <remarks>A Hugging Face checkpoint directory (see <see cref="HfCheckpointDirectory.TryProbe"/>) is one package whose shards are its parts, and the scan does not descend into it.
/// Its draft head is found from the shard index, or, when there is none, from the JSON headers of its safetensors files.
/// Each other <c>.gguf</c> file is one package, except an <c>mmproj</c> GGUF, which is the vision sidecar of the models beside it, and the parts of a llama.cpp split
/// (<c>name-00001-of-000NN.gguf</c>), which together are one package. Hidden directories are skipped, and the scan stops at a fixed depth. A directory that cannot be
/// read is skipped and named in <see cref="TextPackageScan.Problems"/> by its exception type, never its path; the rest of the tree is still listed.
/// A checkpoint without an index has each shard's header read on every scan, so the cost grows with how many such checkpoints there are; nothing is cached.
/// A <c>config.json</c> that does not parse is not reported: its directory is then scanned as plain files.</remarks>
public static class TextPackageDiscovery
{
    private const int MaxDepth = 6;

    private static readonly Regex SplitPart = new(@"^(?<stem>.+)-(?<index>\d{5})-of-(?<total>\d{5})\.gguf$",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);

    /// <summary>The packages under <paramref name="root"/>, ordered by <see cref="TextModelPackage.Id"/>, and the directories that could not be read. A missing directory yields none.</summary>
    public static TextPackageScan Discover(string root)
    {
        string scanRoot = Path.GetFullPath(root);
        List<TextModelPackage> found = [];
        List<string> problems = [];
        if (Directory.Exists(scanRoot))
            Visit(scanRoot, scanRoot, depth: 0, found, problems);
        return new TextPackageScan([.. found.OrderBy(p => p.Id, StringComparer.Ordinal)], problems);
    }

    private static void Visit(string scanRoot, string directory, int depth, List<TextModelPackage> found, List<string> problems)
    {
        IReadOnlyList<TextModelPackage> packages;
        string[] subdirectories;
        try
        {
            HfCheckpointInfo? checkpoint = HfCheckpointDirectory.TryProbe(directory);
            if (checkpoint is not null)
            {
                found.Add(CheckpointPackage(scanRoot, checkpoint));
                return;
            }
            packages = GgufPackages(scanRoot, directory);
            subdirectories = depth >= MaxDepth ? [] : [.. Directory.EnumerateDirectories(directory)
                .Where(d => !Path.GetFileName(d).StartsWith('.')).OrderBy(d => d, StringComparer.Ordinal)];
        }
        catch (Exception exception) when (exception is UnauthorizedAccessException or IOException)
        {
            // One unreadable or vanished directory must not fail the whole listing.
            problems.Add($"{RelativeId(scanRoot, directory)}: skipped ({exception.GetType().Name})");
            return;
        }
        found.AddRange(packages);
        foreach (string sub in subdirectories)
            Visit(scanRoot, sub, depth + 1, found, problems);
    }

    /// <summary>The GGUF packages directly in <paramref name="directory"/>: single files, splits grouped by name, and the <c>mmproj</c> sidecars attached.</summary>
    private static IReadOnlyList<TextModelPackage> GgufPackages(string scanRoot, string directory)
    {
        string[] ggufs = [.. Directory.EnumerateFiles(directory, "*.gguf").OrderBy(f => f, StringComparer.Ordinal)];
        string[] sidecars = [.. ggufs.Where(IsMmproj)];

        List<TextModelPackage> models = [];
        SortedDictionary<(string Stem, int Total), SortedList<int, string>> splits = new();
        foreach (string file in ggufs.Where(f => !IsMmproj(f)))
        {
            Match part = SplitPart.Match(Path.GetFileName(file));
            if (part.Success)
            {
                (string Stem, int Total) key = (part.Groups["stem"].Value, int.Parse(part.Groups["total"].Value, System.Globalization.CultureInfo.InvariantCulture));
                if (!splits.TryGetValue(key, out SortedList<int, string>? parts))
                    splits[key] = parts = new SortedList<int, string>();
                parts[int.Parse(part.Groups["index"].Value, System.Globalization.CultureInfo.InvariantCulture)] = file;
            }
            else
            {
                models.Add(SingleGgufPackage(scanRoot, file));
            }
        }
        ILookup<string, int> totalsByStem = splits.Keys.ToLookup(k => k.Stem, k => k.Total, StringComparer.Ordinal);
        foreach (((string stem, int total), SortedList<int, string> parts) in splits)
        {
            int[] totals = [.. totalsByStem[stem]];
            models.Add(SplitPackage(scanRoot, directory, stem, total, totals, parts));
        }
        return AttachSidecars(models, sidecars);
    }

    private static TextModelPackage CheckpointPackage(string scanRoot, HfCheckpointInfo checkpoint)
    {
        List<string> problems = [];
        bool hasDraft = false;
        HashSet<string> named = new(StringComparer.Ordinal);
        if (checkpoint.IndexPath is not null)
        {
            try
            {
                using JsonDocument index = JsonDocument.Parse(File.ReadAllBytes(checkpoint.IndexPath));
                if (index.RootElement.TryGetProperty("weight_map", out JsonElement map) && map.ValueKind == JsonValueKind.Object)
                {
                    foreach (JsonProperty entry in map.EnumerateObject())
                    {
                        hasDraft |= IsDraftTensor(entry.Name);
                        if (entry.Value.ValueKind == JsonValueKind.String) named.Add(entry.Value.GetString()!);
                    }
                }
            }
            catch (Exception exception) when (exception is JsonException or IOException or UnauthorizedAccessException)
            {
                problems.Add($"shard index unreadable ({exception.GetType().Name})");
            }
        }

        string[] shards = [.. Directory.EnumerateFiles(checkpoint.Root, "*.safetensors").OrderBy(f => f, StringComparer.Ordinal)];
        if (checkpoint.IndexPath is null)
            hasDraft = HeadersNameDraft(shards, problems);
        string[] missing = [.. named.Where(name => !File.Exists(Path.Combine(checkpoint.Root, name))).OrderBy(n => n, StringComparer.Ordinal)];
        if (missing.Length > 0)
            problems.Add($"{missing.Length} shard(s) named by the index are missing: {string.Join(", ", missing)}");

        TextPackageSpeculation speculation = new(false, checkpoint.Flavor == QuantFlavor.Mlx ? TextPackageSpeculation.MlxReason
            : hasDraft ? TextPackageSpeculation.NotWiredReason : TextPackageSpeculation.NoDraftHeadReason);

        return new TextModelPackage(
            RelativeId(scanRoot, checkpoint.Root), checkpoint.Root, TextPackageFormat.SafetensorsShards,
            [.. shards.Select(f => Path.GetFileName(f))], shards.Sum(s => new FileInfo(s).Length),
            checkpoint.ModelType, checkpoint.Flavor?.ToString(), [], speculation, problems);
    }

    /// <summary>Whether any shard's JSON header names a DSpark draft tensor, for a checkpoint with no index to say. A header that cannot be read is named as a problem.</summary>
    private static bool HeadersNameDraft(string[] shards, List<string> problems)
    {
        foreach (string shard in shards)
        {
            try
            {
                SafeTensorHeader header = SafeTensorHeaderReader.Read(shard, SafeTensorHeaderReader.DefaultMaxHeaderBytes, verifyByteLength: false);
                if (header.Tensors.Keys.Any(IsDraftTensor)) return true;
            }
            catch (Exception exception) when (exception is HartsyInferenceException or IOException or UnauthorizedAccessException)
            {
                problems.Add($"{Path.GetFileName(shard)}: safetensors header unreadable ({exception.GetType().Name})");
            }
        }
        return false;
    }

    /// <summary>A DSpark draft (multi-token prediction) tensor.</summary>
    private static bool IsDraftTensor(string name) => name.StartsWith("mtp.", StringComparison.Ordinal);

    private static TextModelPackage SingleGgufPackage(string scanRoot, string file) =>
        new(RelativeId(scanRoot, file), file, TextPackageFormat.Gguf, [Path.GetFileName(file)], new FileInfo(file).Length,
            null, null, [], new(false, TextPackageSpeculation.GgufReason), []);

    /// <summary>The split whose parts share <paramref name="stem"/> and <paramref name="total"/>. When the stem comes with more than one total
    /// (<paramref name="totals"/>), each total is a separate set of parts, so its id also carries the total.</summary>
    private static TextModelPackage SplitPackage(string scanRoot, string directory, string stem, int total, int[] totals, SortedList<int, string> parts)
    {
        List<string> problems = [];
        List<int> missing = [.. Enumerable.Range(1, total).Where(i => !parts.ContainsKey(i))];
        if (missing.Count > 0)
            problems.Add($"missing part(s) {string.Join(", ", missing)} of {total}");
        string name = stem;
        if (totals.Length > 1)
        {
            name = $"{stem}-of-{total:D5}";
            problems.Add($"parts named '{stem}' disagree on the part count ({string.Join(", ", totals)}); each count is listed as its own package");
        }
        // The loader refuses a split file outright (GgufSplitDetector), so a complete split is still not loadable as it stands.
        problems.Add("split GGUF: the loader reads single files only; merge the parts with llama-gguf-split first");

        IReadOnlyList<string> names = [.. parts.Values.Select(f => Path.GetFileName(f))];
        return new TextModelPackage(
            RelativeId(scanRoot, Path.Combine(directory, name)), parts.Values[0], TextPackageFormat.SplitGguf, names,
            parts.Values.Sum(f => new FileInfo(f).Length), null, null, [], new(false, TextPackageSpeculation.GgufReason), problems);
    }

    /// <summary>Attaches a directory's <c>mmproj</c> to the model it belongs to. A lone model takes the first one (any others are named as problems). Several models sharing a directory cannot be told apart by name, so none takes it, and each says so.</summary>
    private static IReadOnlyList<TextModelPackage> AttachSidecars(List<TextModelPackage> models, string[] sidecars)
    {
        if (sidecars.Length == 0 || models.Count == 0) return models;
        string first = Path.GetFileName(sidecars[0]);
        if (models.Count == 1)
        {
            List<string> problems = [.. models[0].Problems];
            if (sidecars.Length > 1)
                problems.Add($"{sidecars.Length} mmproj sidecars beside this model; using {first}");
            return [models[0] with { Sidecars = [new TextPackageSidecar("vision", sidecars[0])], Problems = problems }];
        }
        return [.. models.Select(m => m with
        {
            Problems = [.. m.Problems, $"mmproj {first} is not attached: {models.Count} models share this directory"],
        })];
    }

    private static bool IsMmproj(string file) => Path.GetFileName(file).Contains("mmproj", StringComparison.OrdinalIgnoreCase);

    /// <summary>The package's id: its path relative to the scanned root, with forward slashes, or the root's own name for the root itself.</summary>
    private static string RelativeId(string scanRoot, string path)
    {
        string relative = Path.GetRelativePath(scanRoot, path).Replace(Path.DirectorySeparatorChar, '/');
        return relative == "." ? Path.GetFileName(scanRoot) : relative;
    }
}
