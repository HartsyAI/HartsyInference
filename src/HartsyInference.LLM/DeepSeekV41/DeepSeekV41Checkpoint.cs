using System.Globalization;
using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Quant;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>An opened DeepSeek-V4.1 checkpoint directory: the shard set, the key mapper and the bound quant recipes; it hands out borrowed views and never copies weights.</summary>
/// <remarks>Opening reads headers only. Engram shards are opened pread-only and never mapped. A draft with missing routed experts (the MLX conversion's <c>mtp.2</c>) is reported, its partial experts are kept out of companion binding, and <see cref="RequireDraft"/> refuses it.</remarks>
public sealed class DeepSeekV41Checkpoint : IDisposable
{
    private const int MaxListedProblems = 12;

    private readonly Dictionary<string, string> _sourceByCanonical;
    private readonly HashSet<string> _incompleteDraftKeys;

    private DeepSeekV41Checkpoint(
        HfCheckpointInfo info, DeepSeekV41Config config, QuantFlavor flavor, IHfKeyMapper mapper, ShardedSafeTensorSet shards,
        Dictionary<string, string> sourceByCanonical, QuantBindingSet bindings, DeepSeekV41DraftReport draft,
        HashSet<string> incompleteDraftKeys, DeepSeekV41WeightInventory weights)
    {
        Info = info;
        Config = config;
        Flavor = flavor;
        Mapper = mapper;
        Shards = shards;
        Bindings = bindings;
        Draft = draft;
        Weights = weights;
        _sourceByCanonical = sourceByCanonical;
        _incompleteDraftKeys = incompleteDraftKeys;
    }

    /// <summary>What the directory probe found.</summary>
    public HfCheckpointInfo Info { get; }

    /// <summary>The parsed <c>config.json</c>.</summary>
    public DeepSeekV41Config Config { get; }

    /// <summary>Which producer's companion naming the checkpoint follows.</summary>
    public QuantFlavor Flavor { get; }

    /// <summary>Translates the producer's tensor names to canonical ones.</summary>
    public IHfKeyMapper Mapper { get; }

    /// <summary>The header-first shard set (owned by this checkpoint).</summary>
    public ShardedSafeTensorSet Shards { get; }

    /// <summary>Quant bindings for every complete block-scaled weight, keyed by the producer's own tensor names.</summary>
    public QuantBindingSet Bindings { get; }

    /// <summary>Whether the draft stack is absent, complete or missing routed experts.</summary>
    public DeepSeekV41DraftReport Draft { get; }

    /// <summary>Bytes and tensor counts per memory class over every tensor in the headers, including any incomplete draft experts.</summary>
    public DeepSeekV41WeightInventory Weights { get; }

    /// <summary>Opens the checkpoint at <paramref name="directory"/> from headers alone.</summary>
    /// <exception cref="HartsyInferenceException">The directory is not a DeepSeek-V4.1 checkpoint, is unsharded, has unmapped or colliding keys, or its companions do not bind.</exception>
    public static DeepSeekV41Checkpoint Open(string directory)
    {
        HfCheckpointInfo info = HfCheckpointDirectory.TryProbe(directory)
            ?? throw new HartsyInferenceException($"'{directory}' is not a Hugging Face checkpoint directory (no config.json with model_type, or no safetensors).");
        if (!string.Equals(info.ModelType, DeepSeekV41Config.ModelType, StringComparison.Ordinal))
            throw new HartsyInferenceException($"'{directory}' has model_type '{info.ModelType}', not '{DeepSeekV41Config.ModelType}'.");
        if (info.IndexPath is null)
            throw new HartsyInferenceException($"'{directory}' has no {HfCheckpointDirectory.IndexFileName}; a DeepSeek-V4.1 checkpoint is always sharded.");
        QuantFlavor flavor = info.Flavor
            ?? throw new HartsyInferenceException($"'{info.ConfigPath}' has no recognisable quantization_config; cannot choose companion naming.");
        DeepSeekV41Config config = DeepSeekV41Config.Load(info.ConfigPath);
        IHfKeyMapper mapper = HfKeyMappers.ForSafeTensors(flavor);

        ShardSetOptions options = new() { PreadOnlyShards = EngramShardNames(info.IndexPath, mapper) };
        ShardedSafeTensorSet shards = ShardedSafeTensorSet.OpenIndex(info.Root, options);
        try
        {
            return Build(info, config, flavor, mapper, shards);
        }
        catch
        {
            shards.Dispose();
            throw;
        }
    }

    /// <summary>The canonical key of one routed expert projection; layers at or past the backbone depth are the <c>mtp</c> draft layers.</summary>
    public static string ExpertWeightKey(DeepSeekV41Config config, int layer, int expert, DeepSeekV41ExpertProjection projection)
    {
        ArgumentNullException.ThrowIfNull(config);
        string projectionName = projection.ToString().ToLowerInvariant();
        return string.Create(CultureInfo.InvariantCulture, $"{LayerPrefix(config, layer)}.ffn.experts.{expert}.{projectionName}.weight");
    }

    /// <summary>Whether the checkpoint holds a tensor with this canonical name.</summary>
    public bool HasWeight(string canonicalKey) => TryResolve(canonicalKey, out _);

    /// <summary>Where a tensor's bytes live.</summary>
    public TensorLocation GetLocation(string canonicalKey) => Shards.Inventory[Resolve(canonicalKey)];

    /// <summary>A tensor as a borrowed view of its mapped shard; throws for Engram tables, which are pread-only.</summary>
    public Tensor GetWeight(string canonicalKey) => Shards.GetTensor(Resolve(canonicalKey));

    /// <summary>The weight's quantization recipe, or null when the weight is stored unquantized.</summary>
    public QuantWeightInfo? GetQuant(string canonicalKey)
    {
        string source = Resolve(canonicalKey);
        return Bindings.Bindings.TryGetValue(source, out QuantBinding? binding) ? binding.ToWeightInfo(Shards.GetTensor) : null;
    }

    /// <summary>Access to one layer's routed experts; a draft layer with missing experts is refused.</summary>
    public DeepSeekV41ExpertBank ExpertBank(int layer)
    {
        if (layer < 0 || layer >= Config.TotalLayerCount)
            throw new ArgumentOutOfRangeException(nameof(layer), layer, $"The checkpoint has {Config.TotalLayerCount} layers.");
        bool isDraft = layer >= Config.NumHiddenLayers;
        if (isDraft)
            RequireDraft();
        return new DeepSeekV41ExpertBank(this, layer, isDraft ? Config.DsparkNRoutedExperts : Config.NRoutedExperts);
    }

    /// <summary>Location and pread handles of a layer's Engram table; the bytes are never mapped.</summary>
    public DeepSeekV41EngramTable EngramTable(int layer)
    {
        if (!Config.EngramLayerIds.Contains(layer))
            throw new ArgumentException($"Layer {layer} has no Engram table (Engram layers: {string.Join(", ", Config.EngramLayerIds)}).", nameof(layer));
        string prefix = LayerPrefix(Config, layer);
        TensorLocation embed = GetLocation($"{prefix}.engram.embed.weight");
        TensorLocation? scale = HasWeight($"{prefix}.engram.embed.scale") ? GetLocation($"{prefix}.engram.embed.scale") : null;
        return new DeepSeekV41EngramTable(
            layer, embed, scale, Shards.GetByteSource(embed.Shard), scale is null ? null : Shards.GetByteSource(scale.Shard));
    }

    /// <summary>Throws unless the draft stack is present and complete, naming the missing experts.</summary>
    /// <exception cref="HartsyInferenceException">The draft is absent or incomplete.</exception>
    public void RequireDraft()
    {
        switch (Draft.Status)
        {
            case DeepSeekV41DraftStatus.Complete:
                return;
            case DeepSeekV41DraftStatus.Absent:
                throw new HartsyInferenceException($"This {Flavor} checkpoint carries no draft (mtp) layers, so DSpark speculative decoding is unavailable.");
            default:
                throw new HartsyInferenceException(
                    $"This {Flavor} checkpoint's draft is incomplete, so DSpark speculative decoding is refused: {Draft.DescribeMissing()}.");
        }
    }

    /// <inheritdoc />
    public void Dispose() => Shards.Dispose();

    private static DeepSeekV41Checkpoint Build(
        HfCheckpointInfo info, DeepSeekV41Config config, QuantFlavor flavor, IHfKeyMapper mapper, ShardedSafeTensorSet shards)
    {
        Dictionary<string, string> sourceByCanonical = new(StringComparer.Ordinal);
        List<string> problems = new();
        foreach (string source in shards.Inventory.Keys)
        {
            string? canonical = mapper.MapToCanonical(source);
            if (canonical is null)
            {
                if (!IsStripped(mapper, source))
                    problems.Add($"'{source}' has no canonical name");
                continue;
            }
            if (!sourceByCanonical.TryAdd(canonical, source))
                problems.Add($"'{source}' and '{sourceByCanonical[canonical]}' both map to '{canonical}'");
        }
        if (problems.Count > 0)
            throw new HartsyInferenceException(FormatProblems(info.Root, problems));

        (DeepSeekV41DraftReport draft, HashSet<string> incomplete) = DeepSeekV41DraftScanner.Scan(
            sourceByCanonical.Keys.ToHashSet(StringComparer.Ordinal), config, flavor);
        HashSet<string> incompleteSources = incomplete.Select(key => sourceByCanonical[key]).ToHashSet(StringComparer.Ordinal);
        HashSet<string> mappedSources = sourceByCanonical.Values.ToHashSet(StringComparer.Ordinal);
        Dictionary<string, TensorLocation> bindable = shards.Inventory
            .Where(pair => !incompleteSources.Contains(pair.Key) && mappedSources.Contains(pair.Key))
            .ToDictionary(static pair => pair.Key, static pair => pair.Value, StringComparer.Ordinal);
        QuantBindingSet bindings = QuantCompanionBinder.Bind(bindable, flavor);
        DeepSeekV41WeightInventory weights = DeepSeekV41WeightInventory.Summarize(
            sourceByCanonical.Select(pair => new KeyValuePair<string, long>(pair.Key, shards.Inventory[pair.Value].ByteLength)));
        return new DeepSeekV41Checkpoint(info, config, flavor, mapper, shards, sourceByCanonical, bindings, draft, incomplete, weights);
    }

    private static string[] EngramShardNames(string indexPath, IHfKeyMapper mapper)
    {
        using JsonDocument index = JsonDocument.Parse(File.ReadAllBytes(indexPath));
        if (!index.RootElement.TryGetProperty("weight_map", out JsonElement weightMap) || weightMap.ValueKind != JsonValueKind.Object)
            throw new HartsyInferenceException($"'{indexPath}' has no weight_map object.");
        HashSet<string> names = new(StringComparer.Ordinal);
        foreach (JsonProperty entry in weightMap.EnumerateObject())
        {
            string? canonical = mapper.MapToCanonical(entry.Name);
            if (canonical is not null && DeepSeekV41WeightClassifier.Classify(canonical) == DeepSeekV41WeightClass.Engram
                && entry.Value.GetString() is { } file)
            {
                names.Add(file);
            }
        }
        return names.Order(StringComparer.Ordinal).ToArray();
    }

    private static bool IsStripped(IHfKeyMapper mapper, string source)
    {
        int dot = source.IndexOf('.');
        return mapper.StrippedComponents.Contains(dot < 0 ? source : source[..dot]);
    }

    private static string LayerPrefix(DeepSeekV41Config config, int layer) => layer < config.NumHiddenLayers
        ? string.Create(CultureInfo.InvariantCulture, $"layers.{layer}")
        : string.Create(CultureInfo.InvariantCulture, $"mtp.{layer - config.NumHiddenLayers}");

    private string Resolve(string canonicalKey)
    {
        if (!TryResolve(canonicalKey, out string? source))
        {
            string why = _incompleteDraftKeys.Contains(canonicalKey)
                ? $" (it belongs to an incomplete draft expert: {Draft.DescribeMissing()})"
                : "";
            throw new KeyNotFoundException($"'{canonicalKey}' is not in the checkpoint at {Info.Root}{why}.");
        }
        return source;
    }

    private bool TryResolve(string canonicalKey, out string source)
    {
        ArgumentNullException.ThrowIfNull(canonicalKey);
        if (_incompleteDraftKeys.Contains(canonicalKey))
        {
            source = "";
            return false;
        }
        return _sourceByCanonical.TryGetValue(canonicalKey, out source!);
    }

    private static string FormatProblems(string root, List<string> problems)
    {
        string listed = string.Join("; ", problems.Order(StringComparer.Ordinal).Take(MaxListedProblems));
        string more = problems.Count > MaxListedProblems ? $"; and {problems.Count - MaxListedProblems} more" : "";
        return $"'{root}' has {problems.Count} tensor naming problems: {listed}{more}.";
    }
}
