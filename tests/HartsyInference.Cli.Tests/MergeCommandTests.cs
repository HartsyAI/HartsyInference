using HartsyInference.Cli.Commands;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using Spectre.Console.Cli;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>The <c>merge</c> command through Spectre's parser: arguments bind, the merge runs and exits zero, and bad
/// input exits non-zero before anything is written.</summary>
public sealed class MergeCommandTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-merge-cli-" + Guid.NewGuid().ToString("N"));

    public MergeCommandTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static int Run(params string[] args)
    {
        CommandApp app = new();
        app.Configure(config => config.AddCommand<MergeCommand>("merge"));
        return app.Run(args);
    }

    private void WriteShards()
    {
        WriteShard("model-00001-of-00002.safetensors", "blocks.0.weight", 4, 8);
        WriteShard("model-00002-of-00002.safetensors", "head.weight", 2, 8);
    }

    private void WriteShard(string fileName, string key, long rows, long cols)
    {
        using Tensor f32 = new(new TensorShape(rows, cols), DType.F32);
        using Tensor bf16 = f32.CastTo(DType.BF16);
        SafeTensorsWriter.Save(Path.Combine(_dir, fileName), new Dictionary<string, Tensor> { [key] = bf16 });
    }

    [Fact]
    public void FolderOfShardsMergesThroughTheCommandAndStampsTheModelIdentity()
    {
        WriteShards();
        string output = Path.Combine(_dir, "out", "merged.safetensors");

        int code = Run("merge", _dir, "-o", output, "--model", "iris-z-small", "--overwrite");

        Assert.Equal(0, code);
        CheckpointHeader header = CheckpointHeader.Read(output);
        Assert.Equal(2, header.Descriptors.Count);
        Assert.Equal("iris-z-small", header.Metadata["modelspec.architecture"]);
    }

    [Fact]
    public void ListedShardFilesMergeAndAFolderSiblingIsLeftOut()
    {
        WriteShards();
        WriteShard("conditioning.safetensors", "empty_prompt", 1, 8);
        string output = Path.Combine(_dir, "out", "merged.safetensors");

        int code = Run("merge", Path.Combine(_dir, "model-00001-of-00002.safetensors"),
            Path.Combine(_dir, "model-00002-of-00002.safetensors"), "-o", output, "--overwrite");

        Assert.Equal(0, code);
        CheckpointHeader header = CheckpointHeader.Read(output);
        Assert.Equal(2, header.Descriptors.Count);
        Assert.False(header.Descriptors.ContainsKey("empty_prompt"));
    }

    [Fact]
    public void ExtraMetaIsAppliedAfterTheIdentitySoItWins()
    {
        WriteShards();
        string output = Path.Combine(_dir, "out", "merged.safetensors");

        int code = Run("merge", _dir, "-o", output, "--model", "iris-z-small", "--meta", "modelspec.license=flux-non-commercial", "--overwrite");

        Assert.Equal(0, code);
        Assert.Equal("flux-non-commercial", CheckpointHeader.Read(output).Metadata["modelspec.license"]);
    }

    [Fact]
    public void MissingOutputIsAUsageError()
    {
        WriteShards();

        Assert.Equal(1, Run("merge", _dir));
    }

    [Fact]
    public void UnknownDtypeIsAUsageErrorAndWritesNothing()
    {
        WriteShards();
        string output = Path.Combine(_dir, "out", "merged.safetensors");

        Assert.Equal(1, Run("merge", _dir, "-o", output, "--dtype", "fp8"));
        Assert.False(File.Exists(output));
    }

    [Fact]
    public void UnknownModelIdIsAUsageErrorAndWritesNothing()
    {
        WriteShards();
        string output = Path.Combine(_dir, "out", "merged.safetensors");

        Assert.Equal(1, Run("merge", _dir, "-o", output, "--model", "not-a-model"));
        Assert.False(File.Exists(output));
    }

    [Fact]
    public void MalformedIndexIsAnErrorExitNotAStackTrace()
    {
        string index = Path.Combine(_dir, "broken.safetensors.index.json");
        File.WriteAllText(index, "{ not json");
        string output = Path.Combine(_dir, "out", "merged.safetensors");

        Assert.Equal(1, Run("merge", index, "-o", output));
        Assert.False(File.Exists(output));
    }

    [Fact]
    public void MalformedMetaIsAUsageErrorAndWritesNothing()
    {
        WriteShards();
        string output = Path.Combine(_dir, "out", "merged.safetensors");

        Assert.Equal(1, Run("merge", _dir, "-o", output, "--meta", "no-equals-sign"));
        Assert.False(File.Exists(output));
    }
}
