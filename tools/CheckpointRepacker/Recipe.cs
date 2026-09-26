using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

/// <summary>A declarative repack: which files to read, how their keys map, which tensors to fuse, copy or embed. Field
/// names follow <c>tools/repack/recipe.py</c>, so a recipe reads the same whichever tool runs it.</summary>
internal sealed class Recipe
{
    [JsonPropertyName("name")] public string Name { get; init; } = "";
    [JsonPropertyName("model")] public string? Model { get; init; }
    [JsonPropertyName("variant")] public string? Variant { get; init; }
    [JsonPropertyName("component")] public string? Component { get; init; }
    [JsonPropertyName("source_repo")] public string? SourceRepo { get; init; }
    [JsonPropertyName("dtype")] public string? Dtype { get; init; }
    [JsonPropertyName("components")] public List<RecipeComponent> Components { get; init; } = [];
    [JsonPropertyName("fuse")] public List<RecipeFuse> Fuse { get; init; } = [];
    [JsonPropertyName("copy")] public List<RecipeCopy> Copy { get; init; } = [];
    [JsonPropertyName("drop")] public List<string> Drop { get; init; } = [];
    [JsonPropertyName("embed")] public List<RecipeEmbed> Embed { get; init; } = [];
    [JsonPropertyName("metadata")] public Dictionary<string, string> Metadata { get; init; } = [];

    public static Recipe Load(string path) =>
        JsonSerializer.Deserialize<Recipe>(File.ReadAllText(path), new JsonSerializerOptions { ReadCommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true })
        ?? throw new InvalidDataException($"'{path}' is not a recipe.");

    /// <summary>Opens every component and applies the recipe. Returned tensors are mmap views or owned tensors; keep the
    /// loaders and the owned list alive until the write finishes.</summary>
    public List<KeyValuePair<string, Tensor>> Build(string sourceRoot, List<SafeTensorsLoader> loaders, List<Tensor> owned, List<string> sources,
        Action<string> log)
    {
        List<KeyValuePair<string, Tensor>> tensors = [];
        foreach (RecipeComponent component in Components)
        {
            string path = Path.GetFullPath(Path.Combine(sourceRoot, component.SourceFile));
            (List<KeyValuePair<string, Tensor>> opened, List<SafeTensorsLoader> ls) = SafeTensorsMerger.Open(path);
            loaders.AddRange(ls);
            sources.AddRange(ls.Select(l => l.FilePath));
            List<(Regex From, string To)> renames = [.. component.Renames.Select(r => (new Regex(r[0], RegexOptions.CultureInvariant), r[1]))];
            foreach ((string key, Tensor tensor) in opened)
            {
                string mapped = key;
                foreach ((Regex from, string to) in renames)
                {
                    if (from.IsMatch(mapped))
                    {
                        mapped = from.Replace(mapped, to, 1);
                        break;
                    }
                }
                tensors.Add(new(component.Prefix + mapped, tensor));
            }
            log($"  {component.SourceFile}: {opened.Count} tensors" + (component.Prefix.Length > 0 ? $" under '{component.Prefix}'" : ""));
        }
        Dictionary<string, Tensor> byKey = new(StringComparer.Ordinal);
        foreach ((string key, Tensor tensor) in tensors)
        {
            if (!byKey.TryAdd(key, tensor))
                throw new InvalidDataException($"Two components produce '{key}'; add a prefix or a rename.");
        }
        foreach (RecipeFuse fuse in Fuse)
        {
            int fused = ApplyFuse(fuse, byKey, owned);
            if (fused == 0)
                throw new InvalidDataException($"Fuse into '{fuse.Into}' matched nothing; check its 'from' patterns against the renamed keys.");
            log($"  fused {fused} x [{string.Join(", ", fuse.From)}] -> {fuse.Into}");
        }
        foreach (RecipeCopy copy in Copy)
        {
            if (!byKey.TryGetValue(copy.From, out Tensor? source))
                throw new InvalidDataException($"Copy source '{copy.From}' does not exist.");
            if (!byKey.TryAdd(copy.To, source))
                throw new InvalidDataException($"Copy target '{copy.To}' already exists.");
            log($"  copied {copy.From} -> {copy.To}");
        }
        foreach (string pattern in Drop)
        {
            Regex drop = new("^" + Regex.Escape(pattern).Replace("\\*", ".*").Replace("\\?", ".") + "$");
            int removed = byKey.Keys.Where(k => drop.IsMatch(k)).ToList().Count(k => byKey.Remove(k));
            if (removed == 0)
                throw new InvalidDataException($"Drop '{pattern}' matched nothing.");
            log($"  dropped {removed} x {pattern}");
        }
        foreach (RecipeEmbed embed in Embed)
        {
            byte[] bytes;
            string file = Path.GetFullPath(Path.Combine(sourceRoot, embed.TokenizerFromTiktoken ?? embed.File
                ?? throw new InvalidDataException($"Embed '{embed.Key}' names no file.")));
            sources.Add(file);
            bytes = embed.TokenizerFromTiktoken is not null
                ? TiktokenConverter.ToHuggingFaceJson(file, embed.Pattern ?? throw new InvalidDataException($"Embed '{embed.Key}' needs a 'pattern'."),
                    embed.Normalizer is "none" ? null : embed.Normalizer ?? "NFC")
                : File.ReadAllBytes(file);
            Tensor blob = new(new TensorShape(bytes.Length), DType.U8);
            unsafe
            {
                fixed (byte* src = bytes)
                    Buffer.MemoryCopy(src, (void*)blob.DataPointer, bytes.Length, bytes.Length);
            }
            owned.Add(blob);
            if (!byKey.TryAdd(embed.Key, blob))
                throw new InvalidDataException($"Embed target '{embed.Key}' already exists.");
            log($"  embedded {Path.GetFileName(file)} as {embed.Key} ({bytes.Length} bytes{(embed.TokenizerFromTiktoken is null ? "" : ", tokenizer JSON")})");
        }
        return [.. byKey];
    }

    // "{p}" in a pattern is the shared part of the key; each distinct value of it yields one fused tensor, concatenated
    // along the first axis in the order 'from' lists the parts.
    private static unsafe int ApplyFuse(RecipeFuse fuse, Dictionary<string, Tensor> byKey, List<Tensor> owned)
    {
        Regex first = new("^" + Regex.Escape(fuse.From[0]).Replace("\\{p}", "(?<p>.+)") + "$");
        List<string> stems = [.. byKey.Keys.Select(k => first.Match(k)).Where(m => m.Success).Select(m => m.Groups["p"].Value)];
        foreach (string stem in stems)
        {
            Tensor[] parts = [.. fuse.From.Select(f => byKey.TryGetValue(f.Replace("{p}", stem), out Tensor? t) ? t
                : throw new InvalidDataException($"Fuse part '{f.Replace("{p}", stem)}' is missing."))];
            DType dtype = parts[0].DType;
            int rank = parts[0].Shape.Rank;
            long rows = 0;
            foreach (Tensor part in parts)
            {
                if (part.DType != dtype || part.Shape.Rank != rank)
                    throw new InvalidDataException($"Fuse parts for '{fuse.Into.Replace("{p}", stem)}' differ in dtype or rank.");
                for (int d = 1; d < rank; d++)
                {
                    if (part.Shape[d] != parts[0].Shape[d])
                        throw new InvalidDataException($"Fuse parts for '{fuse.Into.Replace("{p}", stem)}' differ past the first axis.");
                }
                rows += part.Shape[0];
            }
            long[] dims = new long[rank];
            dims[0] = rows;
            for (int d = 1; d < rank; d++)
                dims[d] = parts[0].Shape[d];
            Tensor fused = new(new TensorShape(dims), dtype);
            byte* dst = (byte*)fused.DataPointer;
            foreach (Tensor part in parts)
            {
                long bytes = Tensor.ComputeByteSize(part.Shape, part.DType);
                Buffer.MemoryCopy((void*)part.DataPointer, dst, bytes, bytes);
                dst += bytes;
            }
            owned.Add(fused);
            foreach (string f in fuse.From)
                byKey.Remove(f.Replace("{p}", stem));
            byKey[fuse.Into.Replace("{p}", stem)] = fused;
        }
        return stems.Count;
    }
}

internal sealed class RecipeComponent
{
    [JsonPropertyName("source_file")] public string SourceFile { get; init; } = "";
    [JsonPropertyName("prefix")] public string Prefix { get; init; } = "";
    /// <summary>[regex, replacement] pairs; the first that matches a key rewrites it, the rest are skipped.</summary>
    [JsonPropertyName("renames")] public List<string[]> Renames { get; init; } = [];
}

internal sealed class RecipeFuse
{
    [JsonPropertyName("into")] public string Into { get; init; } = "";
    [JsonPropertyName("from")] public List<string> From { get; init; } = [];
}

internal sealed class RecipeCopy
{
    [JsonPropertyName("from")] public string From { get; init; } = "";
    [JsonPropertyName("to")] public string To { get; init; } = "";
}

internal sealed class RecipeEmbed
{
    [JsonPropertyName("key")] public string Key { get; init; } = "";
    [JsonPropertyName("file")] public string? File { get; init; }
    [JsonPropertyName("tokenizer_from_tiktoken")] public string? TokenizerFromTiktoken { get; init; }
    [JsonPropertyName("pattern")] public string? Pattern { get; init; }
    [JsonPropertyName("normalizer")] public string? Normalizer { get; init; }
}
