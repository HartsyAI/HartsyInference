using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Quant;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Probing a Hugging Face style directory from its config and file listing alone.</summary>
public sealed class HfCheckpointDirectoryTests : IDisposable
{
    private readonly string _directory = Directory.CreateTempSubdirectory("hf-probe-").FullName;

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    private string Write(string name, string content = "{}")
    {
        string path = Path.Combine(_directory, name);
        File.WriteAllText(path, content);
        return path;
    }

    [Fact]
    public void NullEmptyAndMissingDirectories_AreNotCheckpoints()
    {
        Assert.Null(HfCheckpointDirectory.TryProbe(null));
        Assert.Null(HfCheckpointDirectory.TryProbe(" "));
        Assert.Null(HfCheckpointDirectory.TryProbe(Path.Combine(_directory, "missing")));
        Assert.Null(HfCheckpointDirectory.TryProbe(_directory));
    }

    [Fact]
    public void ConfigWithoutWeights_IsNotACheckpoint()
    {
        Write("config.json", "{\"model_type\":\"llama\"}");

        Assert.Null(HfCheckpointDirectory.TryProbe(_directory));
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("{\"model_type\":7}")]
    [InlineData("{\"model_type\":\"\"}")]
    [InlineData("[1]")]
    [InlineData("{not json")]
    public void ConfigWithoutAUsableModelType_IsNotACheckpoint(string config)
    {
        Write("config.json", config);
        Write("model.safetensors", "x");

        Assert.Null(HfCheckpointDirectory.TryProbe(_directory));
    }

    [Fact]
    public void GgufBesideAStrayConfig_IsNotClaimed()
    {
        Write("config.json", "{\"model_type\":\"llama\"}");
        Write("model.gguf", "GGUF");

        Assert.Null(HfCheckpointDirectory.TryProbe(_directory));
    }

    [Fact]
    public void SingleSafetensorsFile_IsACheckpointWithoutAnIndex()
    {
        Write("config.json", "{\"model_type\":\"llama\"}");
        Write("model.safetensors", "x");

        HfCheckpointInfo? info = HfCheckpointDirectory.TryProbe(_directory);

        Assert.NotNull(info);
        Assert.Equal("llama", info!.ModelType);
        Assert.Null(info.IndexPath);
        Assert.Null(info.TokenizerPath);
        Assert.Null(info.Flavor);
        Assert.Equal(_directory, info.Root);
    }

    [Fact]
    public void ShardIndexAlone_IsEnoughAndPathsAreFull()
    {
        string config = Write("config.json", "{\"model_type\":\"deepseek_v41\"}");
        string index = Write("model.safetensors.index.json");
        string tokenizer = Write("tokenizer.json");

        HfCheckpointInfo info = HfCheckpointDirectory.TryProbe(_directory)!;

        Assert.Equal(config, info.ConfigPath);
        Assert.Equal(index, info.IndexPath);
        Assert.Equal(tokenizer, info.TokenizerPath);
    }

    [Fact]
    public void FlavorComesFromTheQuantizationBlock()
    {
        Write("config.json", "{\"model_type\":\"deepseek_v41\",\"quantization\":{\"group_size\":64,\"bits\":4,\"mode\":\"affine\"}}");
        Write("model.safetensors.index.json");

        Assert.Equal(QuantFlavor.Mlx, HfCheckpointDirectory.TryProbe(_directory)!.Flavor);
    }

    [Fact]
    public void RelativeDirectory_IsReportedAsAFullPath()
    {
        Write("config.json", "{\"model_type\":\"llama\"}");
        Write("model.safetensors", "x");
        string relative = Path.GetRelativePath(Directory.GetCurrentDirectory(), _directory);

        Assert.Equal(_directory, HfCheckpointDirectory.TryProbe(relative)!.Root);
    }
}
