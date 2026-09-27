using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Models;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.ModelAssets.Checkpoints;

/// <summary>A header-only snapshot of a checkpoint (tensor names, file-level metadata and the file names that make it
/// up), read without touching tensor data. A directory bundle merges the headers of every <c>*.safetensors</c> file in
/// it. Read errors yield <see cref="Empty"/> rather than throwing, so a capability query never fails on a bad file.</summary>
/// <param name="Keys">Every tensor name, as a full load would map it (GGUF names go through <see cref="CheckpointHeader"/>).</param>
/// <param name="Shapes">Each tensor's shape, keyed like <paramref name="Keys"/>.</param>
/// <param name="Metadata">Safetensors <c>__metadata__</c> strings or the GGUF KV block; the first file's value wins for a bundle.</param>
/// <param name="FileNames">The checkpoint file name without extension, or for a directory its leaf name plus every contained file name.</param>
public sealed record CheckpointProbe(IReadOnlySet<string> Keys, IReadOnlyDictionary<string, TensorShape> Shapes,
    IReadOnlyDictionary<string, string> Metadata, IReadOnlyList<string> FileNames)
{
    private const long MaxHeaderBytes = 64L * 1024 * 1024;

    /// <summary>A probe with no keys, metadata or names.</summary>
    public static CheckpointProbe Empty { get; } = new CheckpointProbe(new HashSet<string>(StringComparer.Ordinal),
        new Dictionary<string, TensorShape>(StringComparer.Ordinal), new Dictionary<string, string>(StringComparer.Ordinal), []);

    /// <summary>Reads the probe for a checkpoint file or bundle directory; <see cref="Empty"/> when the path is missing.</summary>
    public static CheckpointProbe Read(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return Empty;
        }
        HashSet<string> keys = new HashSet<string>(StringComparer.Ordinal);
        Dictionary<string, TensorShape> shapes = new Dictionary<string, TensorShape>(StringComparer.Ordinal);
        Dictionary<string, string> metadata = new Dictionary<string, string>(StringComparer.Ordinal);
        List<string> fileNames = new List<string>();
        if (File.Exists(path))
        {
            fileNames.Add(Path.GetFileNameWithoutExtension(path));
            ReadFileInto(path, keys, shapes, metadata);
            return new CheckpointProbe(keys, shapes, metadata, fileNames);
        }
        if (!Directory.Exists(path))
        {
            return Empty;
        }
        fileNames.Add(Path.GetFileName(Path.TrimEndingDirectorySeparator(path)));
        try
        {
            foreach (string file in Directory.EnumerateFiles(path, "*.safetensors", SearchOption.AllDirectories))
            {
                fileNames.Add(Path.GetFileNameWithoutExtension(file));
                ReadFileInto(file, keys, shapes, metadata);
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            Logs.Warning($"[CheckpointProbe] Could not scan '{path}': {ex.Message}");
        }
        return new CheckpointProbe(keys, shapes, metadata, fileNames);
    }

    /// <summary>True when every one of <paramref name="markers"/> is a tensor name in this checkpoint.</summary>
    public bool HasAllKeys(IReadOnlyList<string> markers)
    {
        foreach (string marker in markers)
        {
            if (!Keys.Contains(marker))
            {
                return false;
            }
        }
        return true;
    }

    /// <summary>True when any tensor name contains one of <paramref name="fragments"/> (ordinal).</summary>
    public bool AnyKeyContains(params string[] fragments)
    {
        foreach (string key in Keys)
        {
            foreach (string fragment in fragments)
            {
                if (key.Contains(fragment, StringComparison.Ordinal))
                {
                    return true;
                }
            }
        }
        return false;
    }

    /// <summary>The metadata value for <paramref name="key"/>, or null when absent or blank.</summary>
    public string? GetMetadata(string key) =>
        Metadata.TryGetValue(key, out string? value) && !string.IsNullOrWhiteSpace(value) ? value : null;

    // Safetensors keeps a hand-rolled JSON read so a header-only file still answers; a GGUF goes through
    // CheckpointHeader so its names map exactly as a full load maps them.
    private static void ReadFileInto(string path, HashSet<string> keys, Dictionary<string, TensorShape> shapes,
        Dictionary<string, string> metadata)
    {
        try
        {
            if (!CheckpointSource.TrySniff(path, out ModelFormat format))
            {
                return;
            }
            if (format == ModelFormat.Gguf)
            {
                CheckpointHeader header = CheckpointHeader.Read(path);
                foreach (KeyValuePair<string, SafeTensorDescriptor> descriptor in header.Descriptors)
                {
                    keys.Add(descriptor.Key);
                    shapes.TryAdd(descriptor.Key, descriptor.Value.Shape);
                }
                foreach (KeyValuePair<string, string> entry in header.Metadata)
                {
                    metadata.TryAdd(entry.Key, entry.Value);
                }
                return;
            }
            using FileStream stream = File.OpenRead(path);
            Span<byte> lengthBuffer = stackalloc byte[8];
            stream.ReadExactly(lengthBuffer);
            long headerLength = BinaryPrimitives.ReadInt64LittleEndian(lengthBuffer);
            if (headerLength is <= 0 or > MaxHeaderBytes)
            {
                return;
            }
            byte[] json = new byte[headerLength];
            stream.ReadExactly(json, 0, (int)headerLength);
            using JsonDocument document = JsonDocument.Parse(json);
            foreach (JsonProperty property in document.RootElement.EnumerateObject())
            {
                if (!string.Equals(property.Name, "__metadata__", StringComparison.Ordinal))
                {
                    keys.Add(property.Name);
                    if (property.Value.ValueKind == JsonValueKind.Object
                        && property.Value.TryGetProperty("shape", out JsonElement shape) && shape.ValueKind == JsonValueKind.Array
                        && shape.GetArrayLength() <= TensorShape.MaxRank)
                    {
                        long[] dims = new long[shape.GetArrayLength()];
                        int i = 0;
                        foreach (JsonElement dim in shape.EnumerateArray())
                        {
                            dims[i++] = dim.GetInt64();
                        }
                        shapes.TryAdd(property.Name, new TensorShape(dims));
                    }
                    continue;
                }
                if (property.Value.ValueKind != JsonValueKind.Object)
                {
                    continue;
                }
                foreach (JsonProperty entry in property.Value.EnumerateObject())
                {
                    if (entry.Value.ValueKind == JsonValueKind.String)
                    {
                        metadata.TryAdd(entry.Name, entry.Value.GetString()!);
                    }
                }
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException or InvalidDataException
            or InvalidOperationException or ArgumentException or HartsyInferenceException)
        {
            Logs.Warning($"[CheckpointProbe] Header read failed for '{path}': {ex.Message}");
        }
    }
}
