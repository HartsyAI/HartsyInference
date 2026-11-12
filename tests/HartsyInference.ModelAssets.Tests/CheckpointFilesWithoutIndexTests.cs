using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Which files <see cref="HfCheckpointDirectory.CheckpointFiles"/> returns for one file of a sharded checkpoint
/// when the folder has no shard index, as after a downloader that fetches only weights. The files are empty: only their
/// names matter.</summary>
public sealed class CheckpointFilesWithoutIndexTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-shards-" + Guid.NewGuid().ToString("N"));

    public CheckpointFilesWithoutIndexTests() => Directory.CreateDirectory(_root);

    public void Dispose() => Directory.Delete(_root, recursive: true);

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void AnyShardBringsTheWholeSetInOrder(int picked)
    {
        string[] shards = Touch("transformer-00001-of-00004.safetensors", "transformer-00002-of-00004.safetensors",
            "transformer-00003-of-00004.safetensors", "transformer-00004-of-00004.safetensors");
        Touch("empty_prompt.safetensors");

        IReadOnlyList<string> files = HfCheckpointDirectory.CheckpointFiles(shards[picked - 1]);

        Assert.Equal(shards, files);
    }

    [Fact]
    public void AMissingSiblingIsReportedByName()
    {
        string[] shards = Touch("transformer-00001-of-00003.safetensors", "transformer-00003-of-00003.safetensors");

        FileNotFoundException error = Assert.Throws<FileNotFoundException>(() => HfCheckpointDirectory.CheckpointFiles(shards[0]));

        Assert.Contains("transformer-00002-of-00003.safetensors", error.Message);
    }

    [Fact]
    public void AFileNotNamedAsAShardStandsAlone()
    {
        string[] files = Touch("model.safetensors", "transformer-00001-of-00002.safetensors", "transformer-00002-of-00002.safetensors");

        Assert.Equal([files[0]], HfCheckpointDirectory.CheckpointFiles(files[0]));
    }

    [Fact]
    public void AnIndexThatNamesTheFileStillDecidesTheSet()
    {
        string[] shards = Touch("transformer-00001-of-00003.safetensors", "transformer-00002-of-00003.safetensors");
        // The index lists two of the three shards the names promise; it wins, so the missing third is not asked for.
        File.WriteAllText(Path.Combine(_root, "transformer.safetensors.index.json"),
            """{"weight_map":{"a":"transformer-00001-of-00003.safetensors","b":"transformer-00002-of-00003.safetensors"}}""");

        Assert.Equal(shards, HfCheckpointDirectory.CheckpointFiles(shards[1]));
    }

    private string[] Touch(params string[] names)
    {
        string[] paths = new string[names.Length];
        for (int i = 0; i < names.Length; i++)
        {
            paths[i] = Path.Combine(_root, names[i]);
            File.WriteAllBytes(paths[i], []);
        }
        return paths;
    }
}
