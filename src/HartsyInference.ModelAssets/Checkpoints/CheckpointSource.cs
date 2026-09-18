using System.Buffers.Binary;
using HartsyInference.Core.Exceptions;
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
    private int _disposed;

    private CheckpointSource(ModelFormat format, string path, IReadOnlyDictionary<string, Tensor> weights,
        CheckpointHeader header, IDisposable handle, GgufMetadata? gguf, string? architecture)
    {
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

        Span<byte> prologue = stackalloc byte[9];
        long length;
        using (FileStream stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            length = stream.Length;
            if (length < prologue.Length || stream.ReadAtLeast(prologue, prologue.Length, throwOnEndOfStream: false) < prologue.Length)
            {
                throw new UnsupportedModelException(
                    $"Checkpoint '{path}' is {length} bytes — too small to be any known format.");
            }
        }

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(prologue);
        if (magic == GgufMagic)
            return ModelFormat.Gguf;

        long headerLength = BinaryPrimitives.ReadInt64LittleEndian(prologue);
        if (headerLength > 0 && headerLength <= length - 8 && prologue[8] == (byte)'{')
            return ModelFormat.SafeTensors;

        throw new UnsupportedModelException(
            $"Checkpoint '{path}' is neither safetensors nor GGUF: it opens with 0x{magic:X8}, which matches no "
            + "container this engine reads.");
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

    private static CheckpointSource OpenSafeTensors(string path, CheckpointOpenOptions options)
    {
        SafeTensorsLoader loader = new SafeTensorsLoader();
        try
        {
            loader.Load(path);
            Dictionary<string, Tensor> weights = loader.GetAllTensors();
            Dictionary<string, string> metadata = loader.Metadata is null
                ? new Dictionary<string, string>(StringComparer.Ordinal)
                : new Dictionary<string, string>(loader.Metadata, StringComparer.Ordinal);
            CheckpointHeader header = new CheckpointHeader(ModelFormat.SafeTensors,
                new Dictionary<string, SafeTensorDescriptor>(loader.Descriptors, StringComparer.Ordinal), metadata);
            return new CheckpointSource(ModelFormat.SafeTensors, path, Normalize(weights, options), header, loader,
                gguf: null, architecture: null);
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
                ? GgufModelLoader.RelabelRank2ToPyTorchOrder(model.Weights)
                : model.Weights;
            Dictionary<string, Tensor> mutable = new(weights.Count, StringComparer.Ordinal);
            foreach (KeyValuePair<string, Tensor> entry in weights) mutable[entry.Key] = entry.Value;
            return new CheckpointSource(ModelFormat.Gguf, path, Normalize(mutable, options),
                CheckpointHeader.Read(path), model, model.Metadata, model.Architecture);
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
        CheckpointOpenOptions options)
    {
        if (!options.FoldQuantCompanions)
            return weights;
        Dictionary<string, Tensor> folded = Nf4CompanionFold.Apply(weights);
        return CheckpointConvertUtils.ApplyFp8ScaledDequant(folded, options.Nvfp4ToFp8, options.ResidentNvfp4);
    }

    /// <summary>"GGUF" read as a little-endian uint32.</summary>
    private const uint GgufMagic = 0x46554747;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _handle.Dispose();
    }
}
