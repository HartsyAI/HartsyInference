using System.Globalization;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>Strict loader for official and hash-bound converted MiniMax-H3 PDD acceleration adapters.</summary>
public sealed class MiniMaxH3PddAdapter : IDisposable
{
    private static readonly string[] _videoWeightAliases =
    [
        "proj_out.weight", "final_layer.video_out.weight", "final_layer.video_out.diff",
        "diffusion_model.final_layer.video_out.diff",
    ];
    private static readonly string[] _videoBiasAliases =
    [
        "proj_out.bias", "final_layer.video_out.bias", "final_layer.video_out.diff_b",
        "diffusion_model.final_layer.video_out.diff_b",
    ];
    private static readonly string[] _audioWeightAliases =
    [
        "audio_proj_out.weight", "final_layer.audio_out.weight", "final_layer.audio_out.diff",
        "diffusion_model.final_layer.audio_out.diff",
    ];
    private static readonly string[] _audioBiasAliases =
    [
        "audio_proj_out.bias", "final_layer.audio_out.bias", "final_layer.audio_out.diff_b",
        "diffusion_model.final_layer.audio_out.diff_b",
    ];

    private SafeTensorsLoader? _loader;
    private int _disposed;

    private MiniMaxH3PddAdapter()
    {
    }

    /// <summary>Adapter path supplied by the operator.</summary>
    public required string FilePath { get; init; }

    /// <summary>Task family declared by adapter metadata or a built-in hash manifest.</summary>
    public required MiniMaxH3PddTask Task { get; init; }

    /// <summary>Published fine-grid interval count.</summary>
    public required int PddNumSteps { get; init; }

    /// <summary>Smallest trained interval grouping.</summary>
    public required int PddBlockSize { get; init; }

    /// <summary>Low-rank dimension declared or inferred from the adapter.</summary>
    public required int Rank { get; init; }

    /// <summary>Global LoRA alpha used by official files that omit per-target alpha tensors.</summary>
    public required float Alpha { get; init; }

    /// <summary>Unmodified safetensors metadata used by planning, notices, and provenance.</summary>
    public IReadOnlyDictionary<string, string>? Metadata { get; init; }

    /// <summary>Shape-checked parallel video/audio projection bank.</summary>
    public required PddHeadBank HeadBank { get; init; }

    /// <summary>Canonical trunk updates with no skipped source tensors.</summary>
    public required MiniMaxH3PddTrunkConversion Trunk { get; init; }

    /// <summary>Non-owning LoRA view accepted by the existing weight-space merge path.</summary>
    public required LoraFile LoraView { get; init; }

    /// <summary>Raw source video-bank weight retained for local conversion output.</summary>
    public required Tensor VideoHeadWeight { get; init; }

    /// <summary>Raw source video-bank bias retained for local conversion output.</summary>
    public required Tensor VideoHeadBias { get; init; }

    /// <summary>Raw source audio-bank weight retained for local conversion output.</summary>
    public required Tensor AudioHeadWeight { get; init; }

    /// <summary>Raw source audio-bank bias retained for local conversion output.</summary>
    public required Tensor AudioHeadBias { get; init; }

    /// <summary>True when all four PDD bank companions are visible in a safetensors header.</summary>
    public static bool IsPddHeader(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors)
    {
        ArgumentNullException.ThrowIfNull(descriptors);
        return HasAny(descriptors, _videoWeightAliases) && HasAny(descriptors, _videoBiasAliases)
            && HasAny(descriptors, _audioWeightAliases) && HasAny(descriptors, _audioBiasAliases);
    }

    /// <summary>Loads and validates a PDD adapter before the generic LoRA detector can reinterpret its trunk updates.</summary>
    public static MiniMaxH3PddAdapter Load(string filePath, MiniMaxH3PddFormatHint formatHint = MiniMaxH3PddFormatHint.Auto,
        MiniMaxH3PddTask hashBoundTask = MiniMaxH3PddTask.Unknown)
    {
        if (!File.Exists(filePath)) throw new FileNotFoundException("MiniMax-H3 PDD adapter not found.", filePath);
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(filePath);
        try
        {
            if (!IsPddHeader(loader.Descriptors))
                throw new HartsyInferenceException($"'{filePath}' does not contain all four MiniMax-H3 PDD head banks.");

            string videoWeightKey = ResolveOne(loader.Descriptors, _videoWeightAliases, "video head weight");
            string videoBiasKey = ResolveOne(loader.Descriptors, _videoBiasAliases, "video head bias");
            string audioWeightKey = ResolveOne(loader.Descriptors, _audioWeightAliases, "audio head weight");
            string audioBiasKey = ResolveOne(loader.Descriptors, _audioBiasAliases, "audio head bias");
            HashSet<string> headKeys = new(StringComparer.Ordinal)
            {
                videoWeightKey, videoBiasKey, audioWeightKey, audioBiasKey,
            };

            Tensor videoWeight = loader.GetTensor(videoWeightKey);
            Tensor videoBias = loader.GetTensor(videoBiasKey);
            Tensor audioWeight = loader.GetTensor(audioWeightKey);
            Tensor audioBias = loader.GetTensor(audioBiasKey);
            MiniMaxH3PddHeadLayout layout = ResolveLayout(videoWeight, loader.Metadata, formatHint);
            PddHeadBank? bank = null;
            MiniMaxH3PddTrunkConversion? trunk = null;
            try
            {
                bank = new PddHeadBank(videoWeight, videoBias, audioWeight, audioBias, layout);
                int numSteps = ReadInt(loader.Metadata, "pdd_num_steps", MiniMaxH3PddSchedule.PublishedFineSteps);
                int blockSize = ReadInt(loader.Metadata, "pdd_block_size", MiniMaxH3PddSchedule.PublishedBlockSize);
                if (numSteps != bank.StepCount || numSteps != MiniMaxH3PddSchedule.PublishedFineSteps)
                {
                    throw new HartsyInferenceException(
                        $"PDD metadata/head count mismatch: pdd_num_steps={numSteps}, bank={bank.StepCount}.");
                }
                if (blockSize != MiniMaxH3PddSchedule.PublishedBlockSize)
                {
                    throw new HartsyInferenceException(
                        $"Published MiniMax-H3 PDD adapters require pdd_block_size=4; got {blockSize}.");
                }
                float alpha = ReadFloat(loader.Metadata, "lora_alpha", 64.0f);
                Dictionary<string, Tensor> all = loader.GetAllTensors();
                trunk = MiniMaxH3PddKeyConverter.Convert(all, headKeys, alpha);
                int rank = ReadInt(loader.Metadata, "lora_rank", InferRank(trunk.Layers));
                if (trunk.Layers.Any(layer => layer.TargetKey.EndsWith("attn.qkv_proj.weight", StringComparison.Ordinal)
                    ? layer.Rank != rank * 3 : layer.Rank != rank))
                {
                    throw new HartsyInferenceException(
                        $"PDD metadata rank {rank} does not match every converted trunk target.");
                }

                MiniMaxH3PddTask metadataTask = ReadTask(loader.Metadata);
                if (metadataTask != MiniMaxH3PddTask.Unknown && hashBoundTask != MiniMaxH3PddTask.Unknown
                    && metadataTask != hashBoundTask)
                {
                    throw new HartsyInferenceException(
                        $"PDD adapter metadata binds {metadataTask}, but its known hash binds {hashBoundTask}.");
                }
                MiniMaxH3PddTask task = hashBoundTask != MiniMaxH3PddTask.Unknown ? hashBoundTask : metadataTask;
                LoraFile view = trunk.CreateLoraView(filePath, loader.Metadata);
                MiniMaxH3PddAdapter adapter = new MiniMaxH3PddAdapter
                {
                    FilePath = filePath,
                    Task = task,
                    PddNumSteps = numSteps,
                    PddBlockSize = blockSize,
                    Rank = rank,
                    Alpha = alpha,
                    Metadata = loader.Metadata,
                    HeadBank = bank,
                    Trunk = trunk,
                    LoraView = view,
                    VideoHeadWeight = videoWeight,
                    VideoHeadBias = videoBias,
                    AudioHeadWeight = audioWeight,
                    AudioHeadBias = audioBias,
                };
                adapter._loader = loader;
                return adapter;
            }
            catch
            {
                trunk?.Dispose();
                bank?.Dispose();
                throw;
            }
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    private static MiniMaxH3PddHeadLayout ResolveLayout(Tensor videoWeight,
        IReadOnlyDictionary<string, string>? metadata, MiniMaxH3PddFormatHint hint)
    {
        if (videoWeight.Shape.Rank == 3)
        {
            if (hint == MiniMaxH3PddFormatHint.KnownFlattenedOffsets)
                throw new HartsyInferenceException("A flattened-offset profile was applied to a rank-three PDD bank.");
            return MiniMaxH3PddHeadLayout.FullHeads;
        }
        if (videoWeight.Shape.Rank != 2)
            throw new HartsyInferenceException($"PDD video bank must be rank 3 or 2; got {videoWeight.Shape}.");
        if (hint == MiniMaxH3PddFormatHint.KnownFlattenedOffsets)
            return MiniMaxH3PddHeadLayout.BasePlusOffsets;

        string? declared = GetMetadata(metadata, "hartsy.pdd.head_layout")
            ?? GetMetadata(metadata, "pdd_head_layout");
        if (string.Equals(declared, "base_plus_offsets_flat", StringComparison.OrdinalIgnoreCase)
            || string.Equals(declared, "flattened_offsets", StringComparison.OrdinalIgnoreCase))
        {
            return MiniMaxH3PddHeadLayout.BasePlusOffsets;
        }
        throw new HartsyInferenceException(
            "A rank-two PDD bank is semantically ambiguous (complete heads vs base-plus-offset rows). "
            + "Only a built-in known hash or explicit hash-bound head-layout metadata may enable it.");
    }

    private static int InferRank(IReadOnlyList<LoraLayer> layers)
    {
        LoraLayer? plain = layers.FirstOrDefault(layer =>
            !layer.TargetKey.EndsWith("attn.qkv_proj.weight", StringComparison.Ordinal));
        if (plain is null) throw new HartsyInferenceException("PDD adapter contains no ordinary trunk LoRA target.");
        return plain.Rank;
    }

    private static MiniMaxH3PddTask ReadTask(IReadOnlyDictionary<string, string>? metadata)
    {
        string? value = GetMetadata(metadata, "pdd_task") ?? GetMetadata(metadata, "pdd_partition")
            ?? GetMetadata(metadata, "hartsy.pdd.task");
        if (value is null) return MiniMaxH3PddTask.Unknown;
        string normalized = value.Trim().Replace("-", string.Empty, StringComparison.Ordinal)
            .Replace("_", string.Empty, StringComparison.Ordinal).ToLowerInvariant();
        return normalized switch
        {
            "fl2va" => MiniMaxH3PddTask.Fl2Va,
            "ref2va" => MiniMaxH3PddTask.Ref2Va,
            _ => MiniMaxH3PddTask.Unknown,
        };
    }

    private static int ReadInt(IReadOnlyDictionary<string, string>? metadata, string key, int fallback)
    {
        string? value = GetMetadata(metadata, key);
        if (value is null) return fallback;
        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed))
            throw new HartsyInferenceException($"PDD metadata '{key}' is not an integer: '{value}'.");
        return parsed;
    }

    private static float ReadFloat(IReadOnlyDictionary<string, string>? metadata, string key, float fallback)
    {
        string? value = GetMetadata(metadata, key);
        if (value is null) return fallback;
        if (!float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed)
            || !(parsed > 0.0f) || !float.IsFinite(parsed))
        {
            throw new HartsyInferenceException($"PDD metadata '{key}' is not a finite positive number: '{value}'.");
        }
        return parsed;
    }

    private static string? GetMetadata(IReadOnlyDictionary<string, string>? metadata, string key) =>
        metadata is not null && metadata.TryGetValue(key, out string? value) ? value : null;

    private static bool HasAny(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors, string[] aliases) =>
        aliases.Any(descriptors.ContainsKey);

    private static string ResolveOne(IReadOnlyDictionary<string, SafeTensorDescriptor> descriptors,
        string[] aliases, string role)
    {
        string[] found = aliases.Where(descriptors.ContainsKey).ToArray();
        if (found.Length != 1)
        {
            throw new HartsyInferenceException(
                $"PDD adapter must contain exactly one {role}; found {found.Length}: {string.Join(", ", found)}.");
        }
        return found[0];
    }

    /// <summary>Releases converted matrices, bank views, and the safetensors mapping in dependency order.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Trunk.Dispose();
        HeadBank.Dispose();
        _loader?.Dispose();
        _loader = null;
    }
}
