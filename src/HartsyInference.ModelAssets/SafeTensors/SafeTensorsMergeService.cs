using System.Security.Cryptography;
using System.Text;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Metadata;

namespace HartsyInference.ModelAssets.SafeTensors;

/// <summary>One offline merge: shards in, one <c>.safetensors</c> file out, streamed tensor by tensor by
/// <see cref="SafeTensorsMerger"/>.</summary>
public sealed record SafeTensorsMergeJob
{
    /// <summary>What to merge, in order. Each entry is a folder (every <c>*.safetensors</c> in it), a
    /// <c>*.safetensors.index.json</c> (the shards it names), or one <c>.safetensors</c> file. Pass the shard files
    /// themselves when a folder holds other safetensors that must stay out, such as a conditioning file beside the
    /// weights: a folder entry takes all of them.</summary>
    public required IReadOnlyList<string> InputPaths { get; init; }

    /// <summary>File to write. Must end in <c>.safetensors</c>, must not be one of the inputs, and must not sit in a
    /// folder input.</summary>
    public required string OutputPath { get; init; }

    /// <summary>Float cast for every F32, BF16 or F16 tensor; null keeps each tensor's stored dtype.</summary>
    public DType? CastTo { get; init; }

    /// <summary>Identity stamped into the header (<c>modelspec.*</c> and <c>hartsy.*</c>). Null writes none, and SwarmUI
    /// cannot classify such a file, so it is not for publishing.</summary>
    public ArtifactIdentity? Identity { get; init; }

    /// <summary>Upstream repository recorded as provenance, e.g. <c>speridlabs/iris-z</c>.</summary>
    public string? SourceRepo { get; init; }

    /// <summary>Extra header entries, applied after the identity so they win on a clash.</summary>
    public IReadOnlyDictionary<string, string>? Metadata { get; init; }

    /// <summary>Whether an existing output may be replaced.</summary>
    public bool Overwrite { get; init; }
}

/// <summary>What the merged file holds, read back from disk after the write rather than taken from the plan.</summary>
public sealed record SafeTensorsMergeReport
{
    /// <summary>Shard files merged.</summary>
    public required int SourceFileCount { get; init; }

    /// <summary>Tensors in the output header.</summary>
    public required int TensorCount { get; init; }

    /// <summary>Bytes of the output file.</summary>
    public required long OutputBytes { get; init; }

    /// <summary>Lowercase hex SHA-256 of the output's tensor payload, computed by re-reading the file.</summary>
    public required string PayloadSha256 { get; init; }
}

/// <summary>Merges sharded safetensors into one file and checks the result against its inputs. Nothing is dropped
/// silently: the output's tensor count must equal the input count, and the payload digest read back from disk must
/// equal the one the writer computed. A file that fails either check is removed and the merge throws.</summary>
public static class SafeTensorsMergeService
{
    private const string Converter = "HartsyInference.SafeTensorsMergeService";

    /// <summary>Runs <paramref name="job"/>.</summary>
    /// <exception cref="HartsyInferenceException">The request is refused before anything is written.</exception>
    /// <exception cref="InvalidDataException">The inputs share a tensor name, or the written file does not match them.</exception>
    public static SafeTensorsMergeReport Merge(SafeTensorsMergeJob job)
    {
        ArgumentNullException.ThrowIfNull(job);
        if (job.InputPaths.Count == 0)
            throw new HartsyInferenceException("Give at least one input: a folder, a *.safetensors.index.json or a .safetensors file.");
        foreach (string input in job.InputPaths)
        {
            if (!Directory.Exists(input) && !File.Exists(input))
                throw new HartsyInferenceException($"Input not found: {input}");
        }
        if (!string.Equals(Path.GetExtension(job.OutputPath), ".safetensors", StringComparison.OrdinalIgnoreCase))
            throw new HartsyInferenceException($"The output must end in .safetensors: {job.OutputPath}");
        if (job.CastTo is { } target && !SafeTensorsMerger.IsCastable(target))
            throw new HartsyInferenceException($"Cannot cast to {target.Name}; use bf16, fp16 or fp32.");
        if (File.Exists(job.OutputPath) && !job.Overwrite)
            throw new HartsyInferenceException($"'{job.OutputPath}' already exists. Pass --overwrite to replace it.");

        string outputFull = Path.GetFullPath(job.OutputPath);
        string outputFolder = Path.TrimEndingDirectorySeparator(Path.GetDirectoryName(outputFull) ?? string.Empty);
        foreach (string input in job.InputPaths)
        {
            // Checked before any folder is read: an output inside a folder input would be listed as a shard, and a previous
            // merge's file there fails to parse as one, which hides the real problem behind a header error.
            if (Directory.Exists(input)
                && string.Equals(Path.TrimEndingDirectorySeparator(Path.GetFullPath(input)), outputFolder, StringComparison.Ordinal))
            {
                throw new HartsyInferenceException(
                    $"'{job.OutputPath}' sits in the input folder, where it would be read back as a shard. Write the merge to another folder.");
            }
        }

        List<KeyValuePair<string, Tensor>> tensors = [];
        List<SafeTensorsLoader> loaders = [];
        try
        {
            foreach (string input in job.InputPaths)
            {
                (List<KeyValuePair<string, Tensor>> inputTensors, List<SafeTensorsLoader> inputLoaders) = SafeTensorsMerger.Open(input);
                tensors.AddRange(inputTensors);
                loaders.AddRange(inputLoaders);
            }
            if (loaders.Any(loader => string.Equals(Path.GetFullPath(loader.FilePath), outputFull, StringComparison.Ordinal)))
                throw new HartsyInferenceException($"'{job.OutputPath}' is one of the inputs. Write the merge to another path.");

            IReadOnlyList<string> sources = loaders.Select(loader => loader.FilePath).ToList();
            string writerDigest = SafeTensorsMerger.Write(job.OutputPath, tensors, job.CastTo,
                BuildMetadata(job, sources, PrecisionLabel(job.CastTo, tensors)));
            return Verify(job.OutputPath, tensors.Count, sources.Count, writerDigest);
        }
        finally
        {
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
        }
    }

    /// <summary>Reopens the written file and checks its header and payload against what was merged.</summary>
    private static SafeTensorsMergeReport Verify(string path, int expectedTensors, int sourceFiles, string writerDigest)
    {
        int tensorCount;
        string? recordedDigest;
        using (SafeTensorsLoader written = new())
        {
            written.Load(path);
            tensorCount = written.Descriptors.Count;
            recordedDigest = written.Metadata?.GetValueOrDefault(ArtifactMetadata.HashKey);
        }
        string payloadDigest = PayloadSha256(path);
        bool digestsAgree = payloadDigest == writerDigest
            && (string.IsNullOrEmpty(recordedDigest) || recordedDigest == "0x" + payloadDigest);
        if (tensorCount != expectedTensors || !digestsAgree)
        {
            File.Delete(path);
            throw new InvalidDataException(
                $"'{path}' holds {tensorCount} tensors and the merge wrote {expectedTensors}, or its payload digest does not "
                + "match what was written. The output was removed.");
        }
        return new SafeTensorsMergeReport
        {
            SourceFileCount = sourceFiles,
            TensorCount = tensorCount,
            OutputBytes = new FileInfo(path).Length,
            PayloadSha256 = payloadDigest,
        };
    }

    /// <summary>Lowercase hex SHA-256 of everything after the header, read back from <paramref name="path"/>.</summary>
    private static string PayloadSha256(string path)
    {
        using FileStream stream = File.OpenRead(path);
        Span<byte> lengthPrefix = stackalloc byte[8];
        stream.ReadExactly(lengthPrefix);
        stream.Seek(8 + BitConverter.ToInt64(lengthPrefix), SeekOrigin.Begin);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    /// <summary>The precision the file is stored in, named as the repacker names it (<c>bf16</c>, <c>fp16</c>, <c>fp32</c>): the
    /// cast when one is asked for, else the single dtype every input tensor shares, or <c>mixed</c>.</summary>
    private static string PrecisionLabel(DType? cast, IReadOnlyList<KeyValuePair<string, Tensor>> tensors)
    {
        if (cast is { } target)
        {
            return PrecisionName(target);
        }
        HashSet<DType> stored = [.. tensors.Select(tensor => tensor.Value.DType)];
        return stored.Count == 1 ? PrecisionName(stored.Single()) : "mixed";
    }

    private static string PrecisionName(DType dtype)
    {
        if (dtype == DType.BF16)
        {
            return "bf16";
        }
        if (dtype == DType.F16)
        {
            return "fp16";
        }
        if (dtype == DType.F32)
        {
            return "fp32";
        }
        return dtype.Name.ToLowerInvariant();
    }

    private static Dictionary<string, string> BuildMetadata(SafeTensorsMergeJob job, IReadOnlyList<string> sources, string precision)
    {
        Dictionary<string, string> metadata = job.Identity is { } identity
            ? ArtifactMetadata.ForRepack(identity, new ArtifactProvenance
            {
                Converter = Converter,
                Component = ArtifactProvenance.MainComponent,
                SourceRepo = job.SourceRepo,
                SourceFile = string.Join(",", sources.Select(Path.GetFileName)),
                SourceSha256 = CombinedSourceSha256(sources),
                Precision = precision,
            })
            : new Dictionary<string, string>(StringComparer.Ordinal);
        if (job.Metadata is not null)
        {
            foreach ((string key, string value) in job.Metadata)
            {
                metadata[key] = value;
            }
        }
        return metadata;
    }

    /// <summary>The source's own digest for one file; for several, a digest over each file's name and digest, in merge order.</summary>
    private static string? CombinedSourceSha256(IReadOnlyList<string> sources)
    {
        if (sources.Count == 1)
        {
            return ArtifactProvenance.HashFile(sources[0]);
        }
        string listing = string.Join('\n', sources.Select(file => $"{Path.GetFileName(file)}:{ArtifactProvenance.HashFile(file)}"));
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(listing))).ToLowerInvariant();
    }
}
