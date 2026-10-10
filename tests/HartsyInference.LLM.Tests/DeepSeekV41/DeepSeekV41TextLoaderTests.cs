using System.Text.Json.Nodes;
using HartsyInference.Core.Exceptions;
using HartsyInference.Cpu;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.DeepSeekV41;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The Text path's Hugging Face directory hook and the V4.1 branch of <see cref="TextService"/>, driven on the small upstream fixture checkpoint.</summary>
public sealed class DeepSeekV41TextLoaderTests : IDisposable
{
    private readonly string _directory = Directory.CreateTempSubdirectory("dsv41-loader-").FullName;

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    private void WriteFixtureCheckpoint(bool tokenizer = true)
    {
        DeepSeekV41ModelFixtureCheckpoint.Write(_directory);
        if (tokenizer) DeepSeekV41ModelFixtureCheckpoint.WriteTokenizer(_directory);
    }

    private static TextRequest Request(string content = "hi", int maxTokens = 3) => new()
    {
        Messages = [new TextMessage { Role = TextRole.User, Content = content }],
        MaxTokens = maxTokens,
        Greedy = true,
        Device = "cpu",
    };

    private ModelSpec Spec() => new() { Requested = "dsv41-fixture", Modality = Modality.Text, LocalPath = _directory };

    [Fact]
    public void ValidDirectory_LoadsTheModelWithItsTokenizerAndTemplate()
    {
        WriteFixtureCheckpoint();
        HfCheckpointInfo info = HfCheckpointDirectory.TryProbe(_directory)!;
        using CpuBackend backend = new();

        using DeepSeekV41TextModel text = HfTextDirectoryLoader.Load(info, backend);

        Assert.Equal(0, text.Tokenizer.BosId);
        Assert.Equal(1, text.Tokenizer.EosId);
        Assert.Equal("deepseek_v41", text.Template.Name);
        Assert.Equal(DeepSeekV41ModelFixtureCheckpoint.Fx.GetProperty("config").GetProperty("vocab_size").GetInt32(), text.Generation.Info.VocabSize);
    }

    [Fact]
    public void ADirectoryWithoutATokenizer_IsRefusedByName()
    {
        WriteFixtureCheckpoint(tokenizer: false);
        HfCheckpointInfo info = HfCheckpointDirectory.TryProbe(_directory)!;
        using CpuBackend backend = new();

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => HfTextDirectoryLoader.Load(info, backend));

        Assert.Contains("tokenizer.json", error.Message);
    }

    [Fact]
    public void ATokenizerWhoseSpecialIdsDisagreeWithConfig_IsRefused()
    {
        WriteFixtureCheckpoint();
        string config = Path.Combine(_directory, "config.json");
        JsonObject root = JsonNode.Parse(File.ReadAllText(config))!.AsObject();
        root["bos_token_id"] = 2;
        File.WriteAllText(config, root.ToJsonString());
        HfCheckpointInfo info = HfCheckpointDirectory.TryProbe(_directory)!;
        using CpuBackend backend = new();

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => HfTextDirectoryLoader.Load(info, backend));

        Assert.Contains("begin-of-sentence", error.Message);
    }

    [Fact]
    public async Task TextService_Generates_Greedily_Through_The_Shared_Pipeline_And_Reuses_The_Loaded_Slot()
    {
        WriteFixtureCheckpoint();
        using InferenceEngine engine = new("cpu", 0);

        TextResult first = await engine.Text.GenerateAsync(Spec(), Request());
        TextResult second = await engine.Text.GenerateAsync(Spec(), Request());

        Assert.True(first.CompletionTokens is >= 1 and <= 3, $"completion tokens: {first.CompletionTokens}");
        Assert.True(first.PromptTokens > 0);
        Assert.Equal(first.Text, second.Text);
        Assert.Equal(first.CompletionTokens, second.CompletionTokens);
        Assert.True(engine.Text.Unload());
    }

    [Fact]
    public void TextService_CountTokens_Uses_The_Checkpoints_Tokenizer()
    {
        WriteFixtureCheckpoint();
        using InferenceEngine engine = new("cpu", 0);

        Assert.Equal(1, engine.Text.CountTokens(Spec(), "hi"));
    }

    [Fact]
    public async Task TextService_Stream_Ends_With_Its_Token_Usage_After_The_Stop_Reason()
    {
        WriteFixtureCheckpoint();
        using InferenceEngine engine = new("cpu", 0);

        List<TextChunk> chunks = [];
        await foreach (TextChunk chunk in engine.Text.StreamAsync(Spec(), Request()))
            chunks.Add(chunk);

        // The chat route's stream_options.include_usage frame is built from this last chunk.
        Assert.Equal(TextChunkKind.StopReason, chunks[^2].Kind);
        Assert.Equal(TextChunkKind.Usage, chunks[^1].Kind);
        Assert.True(chunks[^1].Usage is { PromptTokens: > 0, CompletionTokens: >= 1 and <= 3 }, $"usage: {chunks[^1].Usage}");
    }

    [Fact]
    public async Task TextService_Refuses_A_Request_Longer_Than_The_Loaded_Sequence_Limit()
    {
        WriteFixtureCheckpoint();
        using InferenceEngine engine = new("cpu", 0);

        ArgumentException error = await Assert.ThrowsAnyAsync<ArgumentException>(
            () => engine.Text.GenerateAsync(Spec(), Request(maxTokens: HfTextDirectoryLoader.MaxSequenceTokens)));

        Assert.Contains(HfTextDirectoryLoader.MaxSequenceTokens.ToString(), error.Message);
    }
}
