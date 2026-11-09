using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.Diffusion.Models.TextEncoders;

/// <summary>MOSS-VL-Realtime's vision tower: the Qwen3-VL ViT (identical blocks, position interpolation and rotary) with
/// MOSS's unified deepstack merger, read from the checkpoint's <c>model.visual.*</c> weights and its
/// <c>vision_config</c>, or from the vision blocks a MOSS-VL GGUF carries. Takes preprocessed patches in merge-block order.</summary>
/// <remarks>Given a backend, the linear weights keep 16-bit storage where that backend runs them (see
/// <see cref="DenseWeightCasts"/>); norms, biases and the position table are F32. Without one, everything is widened to F32.</remarks>
public sealed class MossVlVisionTower : IDisposable
{
    private const string Prefix = "model.visual.";

    private readonly IDisposable? _owner;
    private readonly List<Tensor> _converted;
    private int _disposed;

    private MossVlVisionTower(Qwen3VlVisionEncoder encoder, Qwen3VlVisionConfig config, IDisposable? owner, List<Tensor> converted)
    {
        Encoder = encoder;
        Config = config;
        _owner = owner;
        _converted = converted;
    }

    /// <summary>The tower.</summary>
    public Qwen3VlVisionEncoder Encoder { get; }

    /// <summary>The tower's shape, as read from <c>vision_config</c> or the GGUF metadata.</summary>
    public Qwen3VlVisionConfig Config { get; }

    /// <summary>Reads a MOSS-VL <c>vision_config</c> object: the Qwen3-VL field names, with MOSS's merger layout and its exact GELU.</summary>
    public static Qwen3VlVisionConfig ConfigFromJson(JsonElement visionConfig) =>
        Qwen3VlVisionConfig.FromJson(visionConfig) with { UnifiedDeepstackMerger = true, MergerExactGelu = true };

    /// <summary>Reads the tower's shape from a MOSS-VL GGUF's <c>moss_vl.vision.*</c> metadata.</summary>
    /// <exception cref="InvalidDataException">The file carries no vision metadata.</exception>
    public static Qwen3VlVisionConfig ConfigFromGguf(GgufMetadata metadata)
    {
        const string P = "moss_vl.vision.";
        if (!metadata.ContainsKey(P + "depth"))
            throw new InvalidDataException("The GGUF has no moss_vl.vision.* metadata; it carries no MOSS-VL vision tower.");
        Qwen3VlVisionConfig preset = new();
        return new Qwen3VlVisionConfig
        {
            Depth = metadata.GetInteger(P + "depth", preset.Depth),
            HiddenSize = metadata.GetInteger(P + "embedding_length", preset.HiddenSize),
            NumHeads = metadata.GetInteger(P + "num_heads", preset.NumHeads),
            IntermediateSize = metadata.GetInteger(P + "feed_forward_length", preset.IntermediateSize),
            PatchSize = metadata.GetInteger(P + "patch_size", preset.PatchSize),
            SpatialMergeSize = metadata.GetInteger(P + "spatial_merge_size", preset.SpatialMergeSize),
            TemporalPatchSize = metadata.GetInteger(P + "temporal_patch_size", preset.TemporalPatchSize),
            OutHiddenSize = metadata.GetInteger(P + "out_embedding_length", preset.OutHiddenSize),
            NumPositionEmbeddings = metadata.GetInteger(P + "num_position_embeddings", preset.NumPositionEmbeddings),
            DeepstackVisualIndexes = metadata.GetIntegerArray(P + "deepstack_indexes") ?? preset.DeepstackVisualIndexes,
            UnifiedDeepstackMerger = true,
            MergerExactGelu = true,
        };
    }

    /// <summary>Opens the vision tower of a MOSS-VL checkpoint directory (its <c>config.json</c> and safetensors shards).</summary>
    /// <param name="checkpointDirectory">The checkpoint directory.</param>
    /// <param name="backend">The backend that will run the tower; null widens every weight to F32.</param>
    /// <exception cref="InvalidDataException">The checkpoint holds no <c>model.visual.*</c> weights.</exception>
    public static MossVlVisionTower Open(string checkpointDirectory, IBackend? backend = null)
    {
        string configPath = Path.Combine(checkpointDirectory, HfCheckpointDirectory.ConfigFileName);
        Qwen3VlVisionConfig config;
        using (JsonDocument doc = JsonDocument.Parse(File.ReadAllText(configPath)))
            config = ConfigFromJson(doc.RootElement.GetProperty("vision_config"));

        if (HfQuantCheckpoint.DeclaresQuantization(checkpointDirectory))
            return OpenQuantized(checkpointDirectory, config, backend);
        CheckpointSource source = CheckpointSource.OpenShards(HfCheckpointDirectory.WeightFiles(checkpointDirectory));
        return Build(source.Weights, source, config, backend, ggufLayout: false, checkpointDirectory);
    }

    /// <summary>A quantized release (NF4, FP8, …): only the tower's own <c>model.visual.*</c> weights are read, through
    /// <see cref="HfQuantCheckpoint"/>, so the text model's quant companions are never touched. Plain weights are borrowed; any the
    /// release quantized are decoded to F32 here and cast for the backend by <see cref="Build"/>.</summary>
    private static MossVlVisionTower OpenQuantized(string checkpointDirectory, Qwen3VlVisionConfig config, IBackend? backend)
    {
        HfQuantCheckpoint checkpoint = HfQuantCheckpoint.Open(checkpointDirectory);
        Owners owner = new(checkpoint);
        try
        {
            Dictionary<string, Tensor> weights = new(StringComparer.Ordinal);
            foreach (HfQuantWeight w in checkpoint.Weights.Values.Where(w => w.Name.StartsWith(Prefix, StringComparison.Ordinal)))
            {
                Tensor t = w.Storage == HfWeightStorage.Plain ? checkpoint.Borrow(w.Name) : checkpoint.DequantizeF32(w.Name);
                owner.Add(t);
                weights[w.Name] = t;
            }
            return Build(weights, owner, config, backend, ggufLayout: false, checkpointDirectory);
        }
        catch
        {
            owner.Dispose();
            throw;
        }
    }

    /// <summary>The checkpoint and the tensors read from it, released together (tensors first).</summary>
    private sealed class Owners(IDisposable checkpoint) : IDisposable
    {
        private readonly List<Tensor> _tensors = [];
        private int _disposed;

        public void Add(Tensor tensor) => _tensors.Add(tensor);

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            foreach (Tensor t in _tensors) t.Dispose();
            checkpoint.Dispose();
        }
    }

    /// <summary>Opens the vision tower a MOSS-VL GGUF carries beside its text blocks (<c>v.*</c> and <c>mm.*</c>, mapped to
    /// <c>model.visual.*</c> by <c>MossVlKeyMapper</c>). The shape comes from the file's metadata.</summary>
    /// <param name="ggufPath">The GGUF file.</param>
    /// <param name="backend">The backend that will run the tower; null widens every weight to F32.</param>
    /// <exception cref="InvalidDataException">The file carries no vision metadata or weights.</exception>
    public static MossVlVisionTower OpenGguf(string ggufPath, IBackend? backend = null)
    {
        GgufModelLoader.LoadedGgufModel handle = GgufModelLoader.Load(ggufPath);
        try
        {
            Qwen3VlVisionConfig config = ConfigFromGguf(handle.Metadata);
            return Build(handle.Weights, handle, config, backend, ggufLayout: true, ggufPath);
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }

    /// <summary>Removes the rows a GGUF converter padded into a fused qkv weight or bias. q and k pad each rotary half of a head
    /// (<c>headDim/2</c> rows, then zeros up to <c>paddedHeadDim/2</c>); v pads each whole head. The result is the checkpoint's
    /// <c>[3·heads·headDim, …]</c> in the source dtype (caller owns). Returns <paramref name="qkv"/> itself when it is not padded.</summary>
    /// <exception cref="InvalidDataException">The row count is not <c>3·heads·d</c> for a <c>d ≥ headDim</c>, a padded layout has an odd
    /// width, or a padding row is not zero (the converter laid the padding out some other way).</exception>
    public static unsafe Tensor UnpadQkv(Tensor qkv, int heads, int headDim)
    {
        long rows = qkv.Shape[0];
        long cols = qkv.Shape.Rank == 2 ? qkv.Shape[1] : 1;
        long rowsPerSection = rows / 3;
        long padded = rowsPerSection / heads;
        if (rows % 3 != 0 || rowsPerSection % heads != 0 || padded < headDim)
            throw new InvalidDataException($"Fused qkv has {rows} rows, which is not 3 × {heads} heads × a width ≥ {headDim}.");
        if (padded == headDim) return qkv;
        if (padded % 2 != 0 || headDim % 2 != 0)
            throw new InvalidDataException($"Fused qkv is padded from head width {headDim} to {padded}; the rotary-half layout needs even widths.");
        if (qkv.DType.IsQuantized)
            throw new InvalidDataException("A padded fused qkv must be dequantized before it is un-padded.");

        long rowBytes = cols * qkv.DType.SizeInBytes;
        TensorShape shape = qkv.Shape.Rank == 2 ? new TensorShape(3 * heads * headDim, cols) : new TensorShape(3 * heads * headDim);
        Tensor output = new(shape, qkv.DType);
        byte* src = (byte*)qkv.DataPointer;
        byte* dst = (byte*)output.DataPointer;
        long outRow = 0;
        for (int section = 0; section < 3; section++)
        {
            // q and k: the rotary halves are padded separately; v: the whole head.
            long block = section < 2 ? headDim / 2 : headDim;
            long paddedBlock = section < 2 ? padded / 2 : padded;
            long blocks = heads * headDim / block;
            for (long b = 0; b < blocks; b++)
            {
                long srcRow = section * rowsPerSection + b * paddedBlock;
                long bytes = block * rowBytes;
                Buffer.MemoryCopy(src + srcRow * rowBytes, dst + outRow * rowBytes, bytes, bytes);
                outRow += block;
                // The dropped rows must be the converter's zeros; anything else means a different padding layout, which would
                // otherwise load silently and give wrong features.
                long padBytes = (paddedBlock - block) * rowBytes;
                if (!new ReadOnlySpan<byte>(src + (srcRow + block) * rowBytes, (int)padBytes).ContainsAnyExcept((byte)0))
                    continue;
                output.Dispose();
                throw new InvalidDataException($"Fused qkv padding rows {srcRow + block}..{srcRow + paddedBlock - 1} are not zero; "
                    + "this converter pads differently from the layout the loader removes.");
            }
        }
        return output;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Encoder.Dispose();
        foreach (Tensor t in _converted) t.Dispose();
        _owner?.Dispose();
    }

    private static MossVlVisionTower Build(IReadOnlyDictionary<string, Tensor> weights, IDisposable owner, Qwen3VlVisionConfig config,
        IBackend? backend, bool ggufLayout, string origin)
    {
        List<Tensor> converted = [];
        try
        {
            Dictionary<string, Tensor> bare = new(StringComparer.Ordinal);
            foreach ((string key, Tensor value) in weights)
            {
                if (!key.StartsWith(Prefix, StringComparison.Ordinal))
                    continue;
                string name = key[Prefix.Length..];
                Tensor t = value;
                // A GGUF lays rank-2 tensors out [in, out]; the bytes are already the checkpoint's row-major [out, in].
                if (ggufLayout && t.Shape.Rank == 2)
                    t = t.Reshape(new TensorShape((int)t.Shape[1], (int)t.Shape[0]));
                if (name.StartsWith("patch_embed.proj.weight.", StringComparison.Ordinal))
                    throw new NotSupportedException($"'{origin}' splits the patch embedding over temporal kernels ({key}); this loader reads one kernel.");
                // A quantized vision weight decodes straight to what the backend keeps: F32 on the host or CPU, F16 on a device.
                if (t.DType.IsQuantized)
                    t = Own(GgufDequantizer.Dequantize(t, backend is null or Core.Backends.Cpu.CpuBackend ? DType.F32 : DType.F16), t, converted);
                if (ggufLayout && (name.EndsWith(".attn.qkv.weight", StringComparison.Ordinal) || name.EndsWith(".attn.qkv.bias", StringComparison.Ordinal)))
                    t = Own(UnpadQkv(t, config.NumHeads, config.HiddenSize / config.NumHeads), t, converted);
                bare[name] = Own(Store(name, t, backend), t, converted);
            }
            if (!bare.ContainsKey("patch_embed.proj.weight"))
                throw new InvalidDataException($"'{origin}' holds no {Prefix}* vision weights.");

            Qwen3VlVisionEncoder encoder = new(config);
            encoder.LoadWeights(bare);
            return new MossVlVisionTower(encoder, config, owner, converted);
        }
        catch
        {
            foreach (Tensor t in converted) t.Dispose();
            owner.Dispose();
            throw;
        }
    }

    /// <summary>The storage one tower tensor keeps: the linear weights follow the backend, the patch embedding is left for the
    /// encoder to reshape, and everything else (norms, biases, position table) is F32.</summary>
    private static Tensor Store(string name, Tensor t, IBackend? backend)
    {
        if (name == "patch_embed.proj.weight") return t;
        if (t.Shape.Rank == 2 && name != "pos_embed.weight") return DenseWeightCasts.ForBackend(t, backend, Prefix + name);
        return TensorCasts.EnsureF32(t);
    }

    /// <summary>Records <paramref name="result"/> as owned when it is a new tensor, and frees <paramref name="source"/> when it was the last
    /// intermediate this load made (a dequantized or un-padded copy the result replaces); returns <paramref name="result"/>. A mapped
    /// checkpoint tensor or a reshape view is never in <paramref name="converted"/>, so it is never freed here.</summary>
    private static Tensor Own(Tensor result, Tensor source, List<Tensor> converted)
    {
        if (ReferenceEquals(result, source)) return result;
        if (converted.Count > 0 && ReferenceEquals(converted[^1], source))
        {
            source.Dispose();
            converted.RemoveAt(converted.Count - 1);
        }
        converted.Add(result);
        return result;
    }
}
