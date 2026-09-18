using System.Globalization;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Models;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>A checkpoint's tensor inventory — every key with its shape and dtype — plus its file-level metadata, read
/// without mapping a single byte of weight data.</summary>
/// <remarks><para>This is what a planner needs: which tensors a file contains and what shape they are, to decide what
/// the file <i>is</i> before anything commits to loading it. Reading the header of a 20 GB checkpoint costs a few
/// hundred microseconds; loading it costs minutes and a disk's worth of page cache.</para>
/// <para>GGUF headers are normalized to look like safetensors ones — engine key names, <c>[out, in]</c> matrix order —
/// so a caller inspecting shapes never has to know which container it got. The dtypes stay honest: a Q4_K tensor
/// reports <see cref="DType.Q4_K"/>, which is how <see cref="DominantQuantName"/> can name what a file is quantized
/// to.</para></remarks>
/// <param name="Format">The container the header was read from.</param>
/// <param name="Descriptors">Every tensor in the file, keyed by its engine-facing name.</param>
/// <param name="Metadata">The file's own metadata: safetensors' <c>__metadata__</c> strings, or the GGUF KV block rendered as strings.</param>
public sealed record CheckpointHeader(
    ModelFormat Format,
    IReadOnlyDictionary<string, SafeTensorDescriptor> Descriptors,
    IReadOnlyDictionary<string, string> Metadata)
{
    /// <summary>Reads the header of a safetensors or GGUF checkpoint, sniffing the container by magic bytes.</summary>
    public static CheckpointHeader Read(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        ModelFormat format = CheckpointSource.Sniff(path);
        return format == ModelFormat.Gguf ? ReadGguf(path) : ReadSafeTensors(path);
    }

    /// <summary>The quantized dtype most of this checkpoint's bytes are stored in (<c>Q4_K</c>, <c>F8E4M3</c>, …), or null when nothing is quantized.</summary>
    /// <remarks>Weighted by bytes rather than by tensor count: a GGUF keeps its norms and biases at F32, so a count
    /// would report F32 for a file that is overwhelmingly Q4_K.</remarks>
    public string? DominantQuantName()
    {
        Dictionary<string, long> bytesByDtype = new(StringComparer.Ordinal);
        foreach (SafeTensorDescriptor descriptor in Descriptors.Values)
        {
            DType dtype = descriptor.DType;
            if (!dtype.IsQuantized && dtype != DType.F8E4M3 && dtype != DType.F8E5M2 && dtype != DType.I8)
                continue;
            bytesByDtype.TryGetValue(dtype.Name, out long seen);
            bytesByDtype[dtype.Name] = seen + descriptor.ByteLength;
        }
        string? dominant = null;
        long best = 0;
        foreach (KeyValuePair<string, long> entry in bytesByDtype)
        {
            if (entry.Value > best) (dominant, best) = (entry.Key, entry.Value);
        }
        return dominant;
    }

    private static CheckpointHeader ReadSafeTensors(string path)
    {
        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        Dictionary<string, SafeTensorDescriptor> descriptors =
            new Dictionary<string, SafeTensorDescriptor>(loader.Descriptors, StringComparer.Ordinal);
        Dictionary<string, string> metadata = loader.Metadata is null
            ? new Dictionary<string, string>(StringComparer.Ordinal)
            : new Dictionary<string, string>(loader.Metadata, StringComparer.Ordinal);
        return new CheckpointHeader(ModelFormat.SafeTensors, descriptors, metadata);
    }

    private static CheckpointHeader ReadGguf(string path)
    {
        using GgufLoader loader = new GgufLoader();
        loader.Load(path);
        IGgufKeyMapper mapper = GgufModelLoader.ResolveMapping(loader).Mapper;

        Dictionary<string, SafeTensorDescriptor> descriptors = new(loader.Descriptors.Count, StringComparer.Ordinal);
        foreach (KeyValuePair<string, GgufTensorDescriptor> entry in loader.Descriptors)
        {
            string? key = mapper.MapKey(entry.Key);
            if (key is null)
                continue;
            GgufTensorDescriptor source = entry.Value;
            descriptors[key] = new SafeTensorDescriptor
            {
                Name = key,
                DType = source.DType,
                Shape = ReverseNeOrder(source.Shape),
                DataOffset = source.AbsoluteOffset,
                ByteLength = source.DType.ComputeByteCount(source.Shape.ElementCount),
            };
        }
        return new CheckpointHeader(ModelFormat.Gguf, descriptors, RenderMetadata(loader.Metadata));
    }

    /// <summary>Reverses a GGUF shape from ggml's <c>ne</c> order — fastest axis first — to the engine's.</summary>
    /// <remarks>Every rank reverses, not only matrices: a convolution kernel the engine calls
    /// <c>[out, in, kh, kw]</c> is stored <c>[kw, kh, in, out]</c>, and un-reversing only rank 2 would leave it
    /// declaring its kernel width as its output channel count. The byte layout is untouched either way — this is
    /// metadata, and ggml's row-major data already matches the engine's.</remarks>
    private static TensorShape ReverseNeOrder(TensorShape shape)
    {
        if (shape.Rank < 2) return shape;
        Span<long> dims = stackalloc long[shape.Rank];
        shape.CopyDimsTo(dims);
        dims.Reverse();
        return new TensorShape(dims);
    }

    /// <summary>Renders the GGUF KV block as strings so both containers' metadata reads the same way. Arrays — token vocabularies above all — are reported by length rather than materialized, since a caller comparing header metadata never wants a 150k-entry string join.</summary>
    private static Dictionary<string, string> RenderMetadata(GgufMetadata metadata)
    {
        Dictionary<string, string> rendered = new(metadata.Count, StringComparer.Ordinal);
        foreach (string key in metadata.Keys)
        {
            if (!metadata.TryGetValue(key, out object? value) || value is null)
                continue;
            rendered[key] = value switch
            {
                string text => text,
                bool flag => flag ? "true" : "false",
                Array array => $"[{array.Length} values]",
                IFormattable formattable => formattable.ToString(null, CultureInfo.InvariantCulture),
                _ => value.ToString() ?? string.Empty,
            };
        }
        return rendered;
    }

    /// <summary>Throws when <paramref name="key"/> is absent, naming the file so the caller need not re-wrap.</summary>
    public SafeTensorDescriptor Require(string key, string path)
    {
        if (!Descriptors.TryGetValue(key, out SafeTensorDescriptor? descriptor))
            throw new UnsupportedModelException(
                $"Checkpoint '{Path.GetFileName(path)}' has no tensor '{key}'.", null, Format.ToString());
        return descriptor;
    }
}
