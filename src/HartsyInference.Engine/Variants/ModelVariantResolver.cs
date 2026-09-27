using System.Collections.Concurrent;
using HartsyInference.Core.Logging;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.Engine.Variants;

/// <summary>Resolves which <see cref="ModelVariant"/> of a family a checkpoint is, from the strongest evidence
/// available: the weights, then the caller's hint, then the file's metadata, then (as a logged guess) its file name,
/// then the family default. Results are memoised per file state and hints, and each distinct decision is logged
/// once.</summary>
public static class ModelVariantResolver
{
    /// <summary>SAI ModelSpec key SwarmUI classifies by, and that the engine stamps on artifacts it writes.</summary>
    public const string ArchitectureMetadataKey = "modelspec.architecture";

    /// <summary>The engine's own variant stamp (<c>ArtifactProvenance.ModelId</c>).</summary>
    public const string ModelIdMetadataKey = "hartsy.model_id";

    /// <summary>Prefix marking a filename token that must be absent (<c>!base</c>).</summary>
    public const char ExcludePrefix = '!';

    private static readonly char[] TokenSeparators = ['-', '_', '.', ' '];
    private static readonly ConcurrentDictionary<string, ResolvedModelVariant> Resolved = new(StringComparer.Ordinal);

    /// <summary>Resolves <paramref name="evidence"/> against <paramref name="catalog"/>, reading the checkpoint header once per file state.</summary>
    public static ResolvedModelVariant Resolve(ModelVariantCatalog catalog, ModelVariantEvidence evidence)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        ArgumentNullException.ThrowIfNull(evidence);
        string key = CacheKey(catalog, evidence);
        if (Resolved.TryGetValue(key, out ResolvedModelVariant? cached))
        {
            return cached;
        }
        ResolvedModelVariant result = Classify(catalog, CheckpointProbe.Read(evidence.CheckpointPath), evidence.Hints);
        if (Resolved.TryAdd(key, result))
        {
            Report(result, evidence.CheckpointPath);
        }
        return result;
    }

    /// <summary>The pure decision: no I/O, no logging, no cache.</summary>
    public static ResolvedModelVariant Classify(ModelVariantCatalog catalog, CheckpointProbe probe, IReadOnlyList<string?> hints)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        ArgumentNullException.ThrowIfNull(probe);
        ArgumentNullException.ThrowIfNull(hints);
        (ModelVariant? hinted, string? hint) = FindHinted(catalog, hints);
        foreach (ModelVariant variant in catalog.Variants)
        {
            if (variant.HasStructuralRule && probe.HasAllKeys(variant.StructuralMarkers) && (variant.StructuralMatch?.Invoke(probe) ?? true))
            {
                string evidence = variant.StructuralMarkers.Count > 0
                    ? "tensor " + string.Join(", ", variant.StructuralMarkers) : "weight structure";
                return new ResolvedModelVariant(catalog.FamilyId, variant, ModelVariantSource.Structure, evidence)
                {
                    OverriddenHint = hinted is not null && !ReferenceEquals(hinted, variant) ? hint : null,
                };
            }
        }
        if (hinted is not null)
        {
            return new ResolvedModelVariant(catalog.FamilyId, hinted, ModelVariantSource.CallerHint, $"hint '{hint}'");
        }
        string? architecture = probe.GetMetadata(ArchitectureMetadataKey);
        string? modelId = probe.GetMetadata(ModelIdMetadataKey);
        foreach (ModelVariant variant in catalog.Variants)
        {
            if (variant.StructureRequired)
            {
                continue;
            }
            if (architecture is not null && variant.MetadataClassIds.Contains(architecture, StringComparer.OrdinalIgnoreCase))
            {
                return new ResolvedModelVariant(catalog.FamilyId, variant, ModelVariantSource.Metadata,
                    $"{ArchitectureMetadataKey}={architecture}");
            }
            if (modelId is not null && string.Equals(modelId, variant.Id, StringComparison.OrdinalIgnoreCase))
            {
                return new ResolvedModelVariant(catalog.FamilyId, variant, ModelVariantSource.Metadata, $"{ModelIdMetadataKey}={modelId}");
            }
            if (variant.MetadataMatch?.Invoke(probe) == true)
            {
                return new ResolvedModelVariant(catalog.FamilyId, variant, ModelVariantSource.Metadata, "file metadata");
            }
        }
        foreach (ModelVariant variant in catalog.Variants)
        {
            if (variant.StructureRequired)
            {
                continue;
            }
            foreach (string fileName in probe.FileNames)
            {
                IReadOnlyList<string>? tokens = MatchTokenSet(variant, fileName);
                if (tokens is not null)
                {
                    return new ResolvedModelVariant(catalog.FamilyId, variant, ModelVariantSource.Filename,
                        $"file name '{fileName}' has token(s) {string.Join("+", tokens)}");
                }
            }
        }
        return new ResolvedModelVariant(catalog.FamilyId, catalog.Default, ModelVariantSource.Default, "no variant signal found");
    }

    /// <summary>Whole-token split of a file name on <c>- _ . space</c>; the unit filename evidence is matched in.</summary>
    public static string[] Tokenize(string fileName) => fileName.Split(TokenSeparators, StringSplitOptions.RemoveEmptyEntries);

    private static (ModelVariant? Variant, string? Hint) FindHinted(ModelVariantCatalog catalog, IReadOnlyList<string?> hints)
    {
        foreach (string? rawHint in hints)
        {
            string hint = rawHint?.Trim() ?? "";
            if (hint.Length == 0)
            {
                continue;
            }
            foreach (ModelVariant variant in catalog.Variants)
            {
                if (!variant.StructureRequired && variant.MatchesHint(hint))
                {
                    return (variant, hint);
                }
            }
        }
        return (null, null);
    }

    private static IReadOnlyList<string>? MatchTokenSet(ModelVariant variant, string fileName)
    {
        if (variant.FilenameTokenSets.Count == 0)
        {
            return null;
        }
        HashSet<string> tokens = new HashSet<string>(Tokenize(fileName), StringComparer.OrdinalIgnoreCase);
        foreach (IReadOnlyList<string> set in variant.FilenameTokenSets)
        {
            if (set.Count > 0 && set.All(token => token.StartsWith(ExcludePrefix) ? !tokens.Contains(token[1..]) : tokens.Contains(token)))
            {
                return set;
            }
        }
        return null;
    }

    private static string CacheKey(ModelVariantCatalog catalog, ModelVariantEvidence evidence)
    {
        string path = evidence.CheckpointPath ?? "";
        string state = "";
        if (File.Exists(path))
        {
            FileInfo info = new FileInfo(path);
            state = $"{info.Length}@{info.LastWriteTimeUtc.Ticks}";
        }
        else if (Directory.Exists(path))
        {
            state = $"dir@{Directory.GetLastWriteTimeUtc(path).Ticks}";
        }
        return $"{catalog.FamilyId}|{path}|{state}|{string.Join("|", evidence.Hints)}";
    }

    private static void Report(ResolvedModelVariant result, string? checkpointPath)
    {
        string file = string.IsNullOrWhiteSpace(checkpointPath) ? "(no checkpoint)" : Path.GetFileName(Path.TrimEndingDirectorySeparator(checkpointPath));
        string message = $"[ModelVariant] {result.FamilyId} '{file}' → {result.Variant.DisplayName} [{result.Id}] from {result.Source}: {result.Evidence}.";
        if (result.OverriddenHint is not null)
        {
            Logs.Warning($"{message} The weights contradict the requested '{result.OverriddenHint}', so the weights win.");
            return;
        }
        if (result.Source == ModelVariantSource.Filename)
        {
            Logs.Warning($"{message} The file name is only a guess: set the model's class in SwarmUI, pass "
                + $"-m {result.FamilyId}:<variant>, or stamp {ArchitectureMetadataKey} to make it definitive.");
            return;
        }
        Logs.Info(message);
    }
}
