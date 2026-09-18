using System.Buffers.Binary;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Memory;
using HartsyInference.Core.Models;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Nf4;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>An open checkpoint, presented as one weight dictionary regardless of the container it came from.</summary>
/// <remarks><para>Quantized-checkpoint support used to be wired per architecture: a recipe that wanted GGUF took a
/// second constructor parameter and a second code path, so a format worked for the four models somebody had wired it
/// into and failed everywhere else. Nothing about a container is architecture-specific, though — GGUF diffusion files
/// use the same tensor names as the safetensors build they were converted from, so once the keys are mapped, the shapes
/// relabelled and the quantization companions folded, an architecture converter cannot tell the two apart. That is the
/// whole of this type: do those three things once, and every model gets every format.</para>
/// <para>The container is sniffed from magic bytes, never from the extension — repacks are routinely named
/// <c>.safetensors</c> whatever they hold, and a misread extension fails deep inside a parser rather than here.</para>
/// <para><b>Lifetime:</b> the weights borrow memory-mapped file data. They are valid until this source is disposed, and
/// a caller keeping tensors past that point reads freed pages. A converter's output normally borrows too, so the
/// recipe holds the source for as long as it holds the converted weights.</para></remarks>
public sealed class CheckpointSource : IDisposable
{
    private readonly IDisposable _handle;
    private readonly IReadOnlyList<Tensor> _owned;
    private int _disposed;

    private CheckpointSource(ModelFormat format, string path, IReadOnlyDictionary<string, Tensor> weights,
        CheckpointHeader header, IDisposable handle, GgufMetadata? gguf, string? architecture,
        IReadOnlyList<Tensor> owned)
    {
        _owned = owned;
        Format = format;
        Path = path;
        Weights = weights;
        Header = header;
        Gguf = gguf;
        Architecture = architecture;
        _handle = handle;
    }

    /// <summary>The container this checkpoint was read from.</summary>
    public ModelFormat Format { get; }

    /// <summary>The file this was opened from.</summary>
    public string Path { get; }

    /// <summary>Every weight, under its engine-facing key, borrowing memory-mapped file data.</summary>
    public IReadOnlyDictionary<string, Tensor> Weights { get; }

    /// <summary>The file's tensor inventory and metadata.</summary>
    public CheckpointHeader Header { get; }

    /// <summary>The GGUF key-value block, or null for a safetensors checkpoint.</summary>
    public GgufMetadata? Gguf { get; }

    /// <summary>The architecture a GGUF file declares in <c>general.architecture</c>, or null when it declared none and for safetensors.</summary>
    public string? Architecture { get; }

    /// <summary>Identifies a checkpoint's container from its leading bytes.</summary>
    /// <remarks>GGUF opens with its own 4-byte magic. Safetensors has none: it opens with a little-endian 64-bit header
    /// length followed by a JSON object, so a plausible length that lands on a <c>{</c> is the signature. The false
    /// positive rate on that is negligible for the files that reach here, and the alternative — trusting the extension
    /// — is wrong often enough to matter.</remarks>
    public static ModelFormat Sniff(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Checkpoint '{path}' does not exist.", path);
        if (TrySniff(path, out ModelFormat format))
            return format;
        throw new UnsupportedModelException(
            $"Checkpoint '{path}' is neither safetensors nor GGUF: its leading bytes match no container this "
            + "engine reads, and it may be too small to hold a header at all.");
    }

    /// <summary>Identifies a checkpoint's container from its leading bytes, reporting false instead of throwing for a file that is not one this engine reads.</summary>
    /// <remarks>This is what lets a folder be scanned for checkpoints without the extension deciding: a component
    /// directory holds configs and tokenizers beside its weights, and a repack is as likely to be named
    /// <c>.safetensors</c> as <c>.gguf</c> whatever it holds.</remarks>
    public static bool TrySniff(string path, out ModelFormat format)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        format = default;
        if (!File.Exists(path))
            return false;

        Span<byte> prologue = stackalloc byte[9];
        long length;
        using (FileStream stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            length = stream.Length;
            if (length < prologue.Length || stream.ReadAtLeast(prologue, prologue.Length, throwOnEndOfStream: false) < prologue.Length)
                return false;
        }

        if (BinaryPrimitives.ReadUInt32LittleEndian(prologue) == GgufMagic)
        {
            format = ModelFormat.Gguf;
            return true;
        }

        long headerLength = BinaryPrimitives.ReadInt64LittleEndian(prologue);
        if (headerLength > 0 && headerLength <= length - 8 && prologue[8] == (byte)'{')
        {
            format = ModelFormat.SafeTensors;
            return true;
        }
        return false;
    }

    /// <summary>Opens a checkpoint, normalizing it to engine key names, engine matrix order and folded quantization companions.</summary>
    public static CheckpointSource Open(string path, CheckpointOpenOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        CheckpointOpenOptions resolved = options ?? new CheckpointOpenOptions();
        return Sniff(path) switch
        {
            ModelFormat.Gguf => OpenGguf(path, resolved),
            _ => OpenSafeTensors(path, resolved),
        };
    }

    /// <summary>Opens a multi-shard checkpoint as one source: every shard merged first, companions folded once over the whole.</summary>
    /// <remarks><para>Order is the entire point. Safetensors sharding makes no promise that a weight and its
    /// <c>.weight_scale</c> land in the same file, so folding each shard on its own splits pairs that belong together:
    /// an I8 weight whose scale is in the next shard refuses outright, and an fp8 one keeps the default factor of 1.0
    /// while the shard holding its scale drops it as an unclaimed companion — the silent case, which is a weight
    /// running at <c>1/scale</c>.</para>
    /// <para>So the shards are opened unfolded, merged, and folded once. A later shard wins a duplicate key, matching
    /// what every shard loader in this repo already does.</para></remarks>
    public static CheckpointSource OpenShards(IReadOnlyList<string> paths, CheckpointOpenOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(paths);
        if (paths.Count == 0)
            throw new ArgumentException("A sharded checkpoint needs at least one file.", nameof(paths));
        if (paths.Count == 1)
            return Open(paths[0], options);

        CheckpointOpenOptions resolved = options ?? new CheckpointOpenOptions();
        CheckpointOpenOptions unfolded = resolved with { FoldQuantCompanions = false };
        List<CheckpointSource> shards = new List<CheckpointSource>(paths.Count);
        List<Tensor> owned = new List<Tensor>();
        try
        {
            Dictionary<string, Tensor> merged = new Dictionary<string, Tensor>(StringComparer.Ordinal);
            Dictionary<string, SafeTensorDescriptor> descriptors = new(StringComparer.Ordinal);
            Dictionary<string, string> metadata = new(StringComparer.Ordinal);
            ModelFormat format = ModelFormat.SafeTensors;
            foreach (string path in paths)
            {
                CheckpointSource shard = Open(path, unfolded);
                shards.Add(shard);
                // Two formats in one set is a repack beside the release it replaces, not a shard set.
                if (shards.Count > 1 && shard.Format != format)
                {
                    throw new UnsupportedModelException(
                        $"Checkpoint set mixes containers: '{paths[0]}' is {format} and '{path}' is {shard.Format}. "
                        + "A shard set is one format; open the one you meant to load on its own.");
                }
                format = shard.Format;
                foreach (KeyValuePair<string, Tensor> entry in shard.Weights) merged[entry.Key] = entry.Value;
                foreach (KeyValuePair<string, SafeTensorDescriptor> entry in shard.Header.Descriptors)
                    descriptors[entry.Key] = entry.Value;
                foreach (KeyValuePair<string, string> entry in shard.Header.Metadata) metadata[entry.Key] = entry.Value;
            }

            CheckpointHeader header = new CheckpointHeader(format, descriptors, metadata);
            IReadOnlyDictionary<string, Tensor> weights = Normalize(merged, resolved, owned);
            return new CheckpointSource(format, paths[0], weights, header,
                new CompositeDisposable(shards.ToArray()), shards[0].Gguf, shards[0].Architecture, owned);
        }
        catch
        {
            foreach (Tensor tensor in owned) tensor.Dispose();
            foreach (CheckpointSource shard in shards) shard.Dispose();
            throw;
        }
    }

    private static CheckpointSource OpenSafeTensors(string path, CheckpointOpenOptions options)
    {
        SafeTensorsLoader loader = new SafeTensorsLoader();
        try
        {
            loader.Load(path);
            Dictionary<string, Tensor> weights = loader.GetAllTensors();
            List<Tensor> owned = new List<Tensor>();
            Dictionary<string, string> metadata = loader.Metadata is null
                ? new Dictionary<string, string>(StringComparer.Ordinal)
                : new Dictionary<string, string>(loader.Metadata, StringComparer.Ordinal);
            CheckpointHeader header = new CheckpointHeader(ModelFormat.SafeTensors,
                new Dictionary<string, SafeTensorDescriptor>(loader.Descriptors, StringComparer.Ordinal), metadata);
            return new CheckpointSource(ModelFormat.SafeTensors, path, Normalize(weights, options, owned), header,
                loader, gguf: null, architecture: null, owned);
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    private static CheckpointSource OpenGguf(string path, CheckpointOpenOptions options)
    {
        GgufModelLoader.LoadedGgufModel model = GgufModelLoader.Load(path);
        try
        {
            IReadOnlyDictionary<string, Tensor> weights = options.RelabelGgufRank2
                ? GgufModelLoader.RelabelToPyTorchOrder(model.Weights)
                : model.Weights;
            Dictionary<string, Tensor> mutable = new(weights.Count, StringComparer.Ordinal);
            foreach (KeyValuePair<string, Tensor> entry in weights) mutable[entry.Key] = entry.Value;
            List<Tensor> owned = new List<Tensor>();
            return new CheckpointSource(ModelFormat.Gguf, path, Normalize(mutable, options, owned),
                CheckpointHeader.Read(path), model, model.Metadata, model.Architecture, owned);
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }

    /// <summary>Folds quantization companions onto the weights they describe, so an architecture converter sees a weight dictionary whose scales already ride on the tensors.</summary>
    /// <remarks>Order is load-bearing and is the reason this lives here rather than in each converter: a converter
    /// renames <c>blocks.0.attn.qkv.weight</c> but has no rule for <c>blocks.0.attn.qkv.weight_scale</c>, so folding
    /// after the rename pairs nothing and the scale is dropped — the weight then runs at 1/scale, which is not an error,
    /// just noise. Folding first makes that class of bug unreachable.</remarks>
    private static IReadOnlyDictionary<string, Tensor> Normalize(Dictionary<string, Tensor> weights,
        CheckpointOpenOptions options, List<Tensor> owned)
    {
        if (!options.FoldQuantCompanions)
            return weights;
        try
        {
            // The NF4 pass records each decoded weight as it goes rather than on return, because a malformed companion
            // set anywhere in the file aborts the pass with everything before it already allocated.
            Dictionary<string, Tensor> folded = Nf4CompanionFold.Apply(weights, owned);
            Dictionary<string, Tensor> normalized =
                CheckpointConvertUtils.ApplyFp8ScaledDequant(folded, options.Nvfp4ToFp8, options.ResidentNvfp4);
            // Against `folded`, not `weights`: the NF4 decodes are already owned, and re-adding them here would
            // dispose each one twice.
            CollectAllocations(folded, normalized, owned);
            return normalized;
        }
        catch
        {
            foreach (Tensor tensor in owned) tensor.Dispose();
            owned.Clear();
            throw;
        }
    }

    /// <summary>Records every tensor <see cref="Normalize"/> allocated, so this source frees them with the file it borrowed the rest from.</summary>
    /// <remarks>Most normalized weights are the mapped ones with metadata attached — the same objects, nothing to own.
    /// The exceptions allocate: an NF4 weight is decoded into a new tensor, and an eagerly-unpacked NVFP4 one likewise.
    /// Those are gigabytes on a real checkpoint, and a recipe treats what it gets from here as borrowed, so with nobody
    /// tracking them a failed construction or an unload leaves them alive until a finalizer runs. Identity is the test
    /// because it is the actual question: a tensor that is not one of the originals was made here.</remarks>
    private static void CollectAllocations(Dictionary<string, Tensor> original,
        IReadOnlyDictionary<string, Tensor> normalized, List<Tensor> owned)
    {
        HashSet<Tensor> borrowed = new HashSet<Tensor>(original.Count, ReferenceEqualityComparer.Instance as IEqualityComparer<Tensor>);
        foreach (Tensor tensor in original.Values) borrowed.Add(tensor);
        foreach (Tensor tensor in normalized.Values)
        {
            if (!borrowed.Contains(tensor)) owned.Add(tensor);
        }
    }

    /// <summary>"GGUF" read as a little-endian uint32.</summary>
    private const uint GgufMagic = 0x46554747;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (Tensor tensor in _owned) tensor.Dispose();
        _handle.Dispose();
    }
}
