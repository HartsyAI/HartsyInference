using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Metadata;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Sharded safetensors merged into one file. Synthetic BF16 shards with exactly representable values, so the
/// merged values can be checked against the values that were written.</summary>
public sealed class SafeTensorsMergeServiceTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-merge-" + Guid.NewGuid().ToString("N"));

    public SafeTensorsMergeServiceTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    /// <summary>Values <c>seed + i/4</c>: exact in BF16 for the small counts used here.</summary>
    private static float[] Values(float seed, int count) => Enumerable.Range(0, count).Select(i => seed + i * 0.25f).ToArray();

    private static Tensor Bf16(float[] values, long rows, long cols)
    {
        using Tensor f32 = new(new TensorShape(rows, cols), DType.F32);
        values.AsSpan().CopyTo(f32.AsSpan<float>());
        return f32.CastTo(DType.BF16);
    }

    /// <summary>Writes one shard at <paramref name="relativePath"/> under the test folder.</summary>
    private string WriteShard(string relativePath, params (string Key, float[] Values, long Rows, long Cols)[] tensors)
    {
        Dictionary<string, Tensor> map = new(StringComparer.Ordinal);
        foreach ((string key, float[] values, long rows, long cols) in tensors)
        {
            map[key] = Bf16(values, rows, cols);
        }
        string path = Path.Combine(_dir, relativePath);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        SafeTensorsWriter.Save(path, map);
        foreach (Tensor tensor in map.Values)
        {
            tensor.Dispose();
        }
        return path;
    }

    private static SafeTensorsLoader Reopen(string path)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        return loader;
    }

    [Fact]
    public void FolderOfShardsMergesIntoOneFileWithEveryTensorAndItsValues()
    {
        WriteShard("model-00001-of-00002.safetensors",
            ("blocks.0.weight", Values(1f, 32), 4, 8), ("blocks.0.norm", Values(2f, 8), 1, 8));
        WriteShard("model-00002-of-00002.safetensors", ("head.weight", Values(3f, 16), 2, 8));
        string output = Path.Combine(_dir, "out", "merged.safetensors");

        SafeTensorsMergeReport report = SafeTensorsMergeService.Merge(new SafeTensorsMergeJob { InputPaths = [_dir], OutputPath = output });

        Assert.Equal(2, report.SourceFileCount);
        Assert.Equal(3, report.TensorCount);
        Assert.Equal(new FileInfo(output).Length, report.OutputBytes);
        using SafeTensorsLoader merged = Reopen(output);
        Assert.Equal(new[] { "blocks.0.weight", "blocks.0.norm", "head.weight" }, merged.Descriptors.Keys.ToArray());
        using Tensor head = merged.GetTensor("head.weight").CastTo(DType.F32);
        Assert.Equal(Values(3f, 16), head.AsSpan<float>().ToArray());
    }

    [Fact]
    public void ExplicitShardFilesLeaveOutOtherSafetensorsInTheSameFolder()
    {
        WriteShard("model-00001-of-00002.safetensors", ("a.weight", Values(1f, 8), 1, 8));
        WriteShard("model-00002-of-00002.safetensors", ("b.weight", Values(2f, 8), 1, 8));
        WriteShard("conditioning.safetensors", ("empty_prompt", Values(9f, 8), 1, 8));
        string output = Path.Combine(_dir, "out", "merged.safetensors");

        SafeTensorsMergeReport report = SafeTensorsMergeService.Merge(new SafeTensorsMergeJob
        {
            InputPaths = [Path.Combine(_dir, "model-00001-of-00002.safetensors"), Path.Combine(_dir, "model-00002-of-00002.safetensors")],
            OutputPath = output,
        });

        Assert.Equal(2, report.TensorCount);
        using SafeTensorsLoader merged = Reopen(output);
        Assert.False(merged.Descriptors.ContainsKey("empty_prompt"));
    }

    [Fact]
    public void IndexFileMergesTheShardsItNames()
    {
        WriteShard("shards/model-00001-of-00002.safetensors", ("a.weight", Values(1f, 8), 1, 8));
        WriteShard("shards/model-00002-of-00002.safetensors", ("b.weight", Values(2f, 8), 1, 8));
        string index = Path.Combine(_dir, "shards", "model.safetensors.index.json");
        File.WriteAllText(index, JsonSerializer.Serialize(new
        {
            metadata = new { total_size = 0 },
            weight_map = new Dictionary<string, string>
            {
                ["a.weight"] = "model-00001-of-00002.safetensors",
                ["b.weight"] = "model-00002-of-00002.safetensors",
            },
        }));
        string output = Path.Combine(_dir, "out", "merged.safetensors");

        SafeTensorsMergeReport report = SafeTensorsMergeService.Merge(new SafeTensorsMergeJob { InputPaths = [index], OutputPath = output });

        Assert.Equal(2, report.SourceFileCount);
        Assert.Equal(2, report.TensorCount);
    }

    [Fact]
    public void IdentityFromTheCatalogIsStampedAndTheDigestMatchesThePayloadOnDisk()
    {
        ArtifactIdentity identity = ModelIdentityCatalog.Find("iris-z-small")!;
        WriteShard("model-00001-of-00001.safetensors", ("head.weight", Values(1f, 16), 2, 8));
        string output = Path.Combine(_dir, "out", "merged.safetensors");

        SafeTensorsMergeReport report = SafeTensorsMergeService.Merge(new SafeTensorsMergeJob
        {
            InputPaths = [_dir],
            OutputPath = output,
            Identity = identity,
            SourceRepo = "speridlabs/iris-z",
        });

        using SafeTensorsLoader merged = Reopen(output);
        Assert.Equal("iris-z-small", merged.Metadata!["modelspec.architecture"]);
        Assert.Equal("Iris-Z-small (bf16)", merged.Metadata["modelspec.title"]);
        Assert.Equal("bf16", merged.Metadata["hartsy.precision"]);
        Assert.Equal(identity.License, merged.Metadata["modelspec.license"]);
        Assert.Equal("speridlabs/iris-z", merged.Metadata["hartsy.source_repo"]);
        Assert.Equal("0x" + report.PayloadSha256, merged.Metadata[ArtifactMetadata.HashKey]);
    }

    [Fact]
    public void CastToF16StoresTheFloatTensorsAsF16()
    {
        WriteShard("model-00001-of-00001.safetensors", ("head.weight", Values(1f, 16), 2, 8));
        string output = Path.Combine(_dir, "out", "merged.safetensors");

        SafeTensorsMergeService.Merge(new SafeTensorsMergeJob { InputPaths = [_dir], OutputPath = output, CastTo = DType.F16 });

        using SafeTensorsLoader merged = Reopen(output);
        Assert.Equal(DType.F16, merged.Descriptors["head.weight"].DType);
    }

    [Fact]
    public void ExistingOutputIsRefusedUntilOverwriteIsSet()
    {
        WriteShard("model-00001-of-00001.safetensors", ("head.weight", Values(1f, 16), 2, 8));
        string output = Path.Combine(_dir, "out", "merged.safetensors");
        Directory.CreateDirectory(Path.GetDirectoryName(output)!);
        File.WriteAllText(output, "not a checkpoint");

        HartsyInferenceException refused = Assert.Throws<HartsyInferenceException>(() =>
            SafeTensorsMergeService.Merge(new SafeTensorsMergeJob { InputPaths = [_dir], OutputPath = output }));
        Assert.Contains("--overwrite", refused.Message);

        SafeTensorsMergeService.Merge(new SafeTensorsMergeJob { InputPaths = [_dir], OutputPath = output, Overwrite = true });
        using SafeTensorsLoader merged = Reopen(output);
        Assert.Single(merged.Descriptors);
    }

    [Fact]
    public void OutputInsideTheInputFolderIsRefusedBeforeAnythingIsRead()
    {
        WriteShard("model-00001-of-00001.safetensors", ("head.weight", Values(1f, 16), 2, 8));
        string output = Path.Combine(_dir, "merged.safetensors");
        File.WriteAllText(output, "not a checkpoint");

        HartsyInferenceException refused = Assert.Throws<HartsyInferenceException>(() =>
            SafeTensorsMergeService.Merge(new SafeTensorsMergeJob { InputPaths = [_dir], OutputPath = output, Overwrite = true }));

        Assert.Contains("input folder", refused.Message);
        Assert.Equal("not a checkpoint", File.ReadAllText(output));
    }

    [Fact]
    public void OutputThatIsAnInputIsRefusedAndTheInputSurvives()
    {
        string shard = WriteShard("model-00001-of-00001.safetensors", ("head.weight", Values(1f, 16), 2, 8));
        long before = new FileInfo(shard).Length;

        Assert.Throws<HartsyInferenceException>(() =>
            SafeTensorsMergeService.Merge(new SafeTensorsMergeJob { InputPaths = [shard], OutputPath = shard, Overwrite = true }));

        Assert.Equal(before, new FileInfo(shard).Length);
    }

    [Fact]
    public void TheSameTensorInTwoShardsIsRefusedAndLeavesNoOutput()
    {
        WriteShard("model-00001-of-00002.safetensors", ("head.weight", Values(1f, 16), 2, 8));
        WriteShard("model-00002-of-00002.safetensors", ("head.weight", Values(2f, 16), 2, 8));
        string output = Path.Combine(_dir, "out", "merged.safetensors");

        InvalidDataException refused = Assert.Throws<InvalidDataException>(() =>
            SafeTensorsMergeService.Merge(new SafeTensorsMergeJob { InputPaths = [_dir], OutputPath = output }));
        Assert.Contains("more than one input", refused.Message);
        Assert.False(File.Exists(output));
        Assert.False(File.Exists(output + ".tmp"));
    }

    [Fact]
    public void OutputNotEndingInSafetensorsIsRefused()
    {
        WriteShard("model-00001-of-00001.safetensors", ("head.weight", Values(1f, 16), 2, 8));

        HartsyInferenceException refused = Assert.Throws<HartsyInferenceException>(() =>
            SafeTensorsMergeService.Merge(new SafeTensorsMergeJob { InputPaths = [_dir], OutputPath = Path.Combine(_dir, "out", "merged.gguf") }));
        Assert.Contains(".safetensors", refused.Message);
    }

    [Fact]
    public void MissingInputIsRefusedBeforeAnythingIsWritten()
    {
        string output = Path.Combine(_dir, "out", "merged.safetensors");

        Assert.Throws<HartsyInferenceException>(() =>
            SafeTensorsMergeService.Merge(new SafeTensorsMergeJob { InputPaths = [Path.Combine(_dir, "nope")], OutputPath = output }));
        Assert.False(File.Exists(output));
    }

    [Fact]
    public void SuppliedDigestIsRefusedBeforeAnythingIsWritten()
    {
        WriteShard("model-00001-of-00001.safetensors", ("head.weight", Values(1f, 16), 2, 8));
        string output = Path.Combine(_dir, "out", "merged.safetensors");

        HartsyInferenceException refused = Assert.Throws<HartsyInferenceException>(() =>
            SafeTensorsMergeService.Merge(new SafeTensorsMergeJob
            {
                InputPaths = [_dir],
                OutputPath = output,
                Metadata = new Dictionary<string, string> { [ArtifactMetadata.HashKey] = "0x" + new string('1', 64) },
            }));

        Assert.Contains("computes", refused.Message);
        Assert.False(File.Exists(output));
    }

    [Fact]
    public void NoInputIsRefused()
    {
        Assert.Throws<HartsyInferenceException>(() =>
            SafeTensorsMergeService.Merge(new SafeTensorsMergeJob { InputPaths = [], OutputPath = Path.Combine(_dir, "out", "merged.safetensors") }));
    }
}
