using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.SafeTensors;

/// <summary>Writes tensors gathered from any number of sources (shards, companion files, borrowed mmap views) into one
/// safetensors file, streaming tensor by tensor with an optional float cast — so a 19 GB fp32 sharded release becomes a
/// 9.5 GB bf16 file without holding either in memory.
/// <para>Casts round to nearest even, bit for bit as <c>torch.Tensor.to</c> does, so a file produced here matches one a
/// PyTorch-based repack made from the same source. The engine's host cast truncates and must not be used for this.</para></summary>
public static class SafeTensorsMerger
{
    private const string HashKey = "modelspec.hash_sha256";
    private const int ChunkElements = 1 << 22;

    /// <summary>Whether <paramref name="dtype"/> is a float the merger can cast between (F32, BF16, F16).</summary>
    public static bool IsCastable(DType dtype) => dtype == DType.F32 || dtype == DType.BF16 || dtype == DType.F16;

    /// <summary>Writes <paramref name="tensors"/> in order to <paramref name="outputPath"/>. Float tensors are cast to
    /// <paramref name="castTo"/> when given; integer and byte tensors pass through. An empty
    /// <c>modelspec.hash_sha256</c> in <paramref name="metadata"/> is filled with the payload digest.</summary>
    /// <returns>Lowercase hex SHA-256 of the tensor payload.</returns>
    /// <param name="keepDtype">Keys that keep their stored dtype through a cast, e.g. norms kept in F32 under a BF16 cast.</param>
    public static unsafe string Write(string outputPath, IReadOnlyList<KeyValuePair<string, Tensor>> tensors, DType? castTo = null,
        IReadOnlyDictionary<string, string>? metadata = null, Func<string, bool>? keepDtype = null)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        if (castTo is { } target && !IsCastable(target))
        {
            throw new ArgumentException($"Cannot cast to {target.Name}; supported targets are F32, BF16 and F16.", nameof(castTo));
        }
        HashSet<string> seen = new(StringComparer.Ordinal);
        List<(string Key, Tensor Source, DType OutType, long Bytes)> plan = new(tensors.Count);
        foreach ((string key, Tensor tensor) in tensors)
        {
            if (!seen.Add(key))
            {
                throw new InvalidDataException($"Tensor '{key}' appears in more than one input; merging would lose one of them.");
            }
            DType outType = castTo is { } cast && IsCastable(tensor.DType) && keepDtype?.Invoke(key) != true ? cast : tensor.DType;
            plan.Add((key, tensor, outType, Tensor.ComputeByteSize(tensor.Shape, outType)));
        }
        Dictionary<string, string> meta = metadata is null ? new(StringComparer.Ordinal) : new(metadata, StringComparer.Ordinal);
        bool fillHash = meta.TryGetValue(HashKey, out string? slot) && string.IsNullOrEmpty(slot);
        if (fillHash)
        {
            meta[HashKey] = "0x" + new string('0', 64);
        }
        string? outputDir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(outputDir))
            Directory.CreateDirectory(outputDir);
        string tempPath = outputPath + ".tmp";
        string payloadHash;
        try
        {
            using (FileStream output = new(tempPath, FileMode.Create, FileAccess.ReadWrite, FileShare.None, 1 << 22))
            {
                byte[] header = BuildHeader(plan, meta);
                output.Write(BitConverter.GetBytes((long)header.Length));
                output.Write(header);
                using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                byte[] buffer = new byte[ChunkElements * 4];
                foreach ((string _, Tensor source, DType outType, long outBytes) in plan)
                {
                    if (outType == source.DType)
                    {
                        // Copied by byte count: a block-packed dtype (F4_E2M1, …) has no per-element size.
                        byte* raw = (byte*)source.DataPointer;
                        for (long offset = 0; offset < outBytes; offset += buffer.Length)
                        {
                            ReadOnlySpan<byte> part = new(raw + offset, (int)Math.Min(buffer.Length, outBytes - offset));
                            hash.AppendData(part);
                            output.Write(part);
                        }
                        continue;
                    }
                    long count = source.Shape.ElementCount;
                    byte* src = (byte*)source.DataPointer;
                    int inSize = source.DType.SizeInBytes;
                    int outSize = outType.SizeInBytes;
                    for (long done = 0; done < count; done += ChunkElements)
                    {
                        int n = (int)Math.Min(ChunkElements, count - done);
                        fixed (byte* dst = buffer)
                        {
                            Convert(src + done * inSize, source.DType, dst, outType, n);
                        }
                        ReadOnlySpan<byte> chunk = new(buffer, 0, n * outSize);
                        hash.AppendData(chunk);
                        output.Write(chunk);
                    }
                }
                payloadHash = System.Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
                if (fillHash)
                {
                    meta[HashKey] = "0x" + payloadHash;
                    output.Seek(8, SeekOrigin.Begin);
                    output.Write(BuildHeader(plan, meta));
                }
            }
            File.Move(tempPath, outputPath, overwrite: true);
        }
        catch
        {
            File.Delete(tempPath);
            throw;
        }
        return payloadHash;
    }

    /// <summary>Opens <paramref name="path"/> as merge input: a <c>*.safetensors.index.json</c> (its shards), a folder
    /// (its <c>*.safetensors</c>, by name), or one file. Tensors come back as mmap views in on-disk order, keys prefixed
    /// with <paramref name="prefix"/>; the returned loaders own the views and must outlive the write.</summary>
    public static (List<KeyValuePair<string, Tensor>> Tensors, List<SafeTensorsLoader> Loaders) Open(string path, string prefix = "")
    {
        List<string> files;
        if (Directory.Exists(path))
        {
            files = [.. Directory.EnumerateFiles(path, "*.safetensors").Order(StringComparer.Ordinal)];
            if (files.Count == 0)
                throw new FileNotFoundException($"No .safetensors files in '{path}'.");
        }
        else if (path.EndsWith(".index.json", StringComparison.OrdinalIgnoreCase))
        {
            using JsonDocument index = JsonDocument.Parse(File.ReadAllBytes(path));
            if (!index.RootElement.TryGetProperty("weight_map", out JsonElement weightMap))
                throw new InvalidDataException($"'{path}' has no weight_map.");
            string dir = Path.GetDirectoryName(Path.GetFullPath(path))!;
            files = [.. weightMap.EnumerateObject().Select(e => Path.Combine(dir, e.Value.GetString()!)).Distinct().Order(StringComparer.Ordinal)];
        }
        else
        {
            files = [path];
        }
        List<KeyValuePair<string, Tensor>> tensors = [];
        List<SafeTensorsLoader> loaders = [];
        foreach (string file in files)
        {
            SafeTensorsLoader loader = new();
            loader.Load(file);
            loaders.Add(loader);
            foreach (SafeTensorDescriptor d in loader.Descriptors.Values.OrderBy(d => d.DataOffset))
                tensors.Add(new(prefix + d.Name, loader.GetTensor(d.Name)));
        }
        return (tensors, loaders);
    }

    /// <summary>Converts <paramref name="count"/> floats between F32, BF16 and F16, rounding to nearest even.</summary>
    public static unsafe void Convert(byte* src, DType from, byte* dst, DType to, int count)
    {
        for (int i = 0; i < count; i++)
        {
            float value = from == DType.F32 ? ((float*)src)[i]
                : from == DType.BF16 ? BitConverter.Int32BitsToSingle(((ushort*)src)[i] << 16)
                : (float)((Half*)src)[i];
            if (to == DType.F32)
                ((float*)dst)[i] = value;
            else if (to == DType.BF16)
                ((ushort*)dst)[i] = ToBf16(value);
            else
                ((Half*)dst)[i] = (Half)value;
        }
    }

    /// <summary>F32 → BF16 with round-to-nearest-even; NaN becomes the canonical quiet NaN, as in PyTorch.</summary>
    public static ushort ToBf16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if (float.IsNaN(value))
            return 0x7FC0;
        return (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
    }

    private static byte[] BuildHeader(List<(string Key, Tensor Source, DType OutType, long Bytes)> plan, Dictionary<string, string> metadata)
    {
        JsonObject root = new();
        if (metadata.Count > 0)
        {
            JsonObject meta = new();
            foreach ((string key, string value) in metadata)
                meta[key] = value;
            root["__metadata__"] = meta;
        }
        long offset = 0;
        foreach ((string key, Tensor source, DType outType, long bytes) in plan)
        {
            JsonArray shape = new();
            for (int d = 0; d < source.Shape.Rank; d++)
                shape.Add(source.Shape[d]);
            root[key] = new JsonObject
            {
                ["dtype"] = outType.Name,
                ["shape"] = shape,
                ["data_offsets"] = new JsonArray(offset, offset + bytes),
            };
            offset += bytes;
        }
        return Encoding.UTF8.GetBytes(root.ToJsonString(new JsonSerializerOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }));
    }
}
