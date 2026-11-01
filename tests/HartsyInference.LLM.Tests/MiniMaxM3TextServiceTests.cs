using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>The Text path's MiniMax-M3 branch: a directory in the release's layout (the tiny fixture) is recognised, hosted on the CPU backend and generates, and a directory missing its chat template is refused by name.</summary>
public sealed class MiniMaxM3TextServiceTests : IDisposable
{
    private const int VocabSize = 64;
    private readonly string _directory = Directory.CreateTempSubdirectory("minimax-m3-text-").FullName;

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    private static TextRequest Request(string content) => new()
    {
        Messages = [new TextMessage { Role = TextRole.User, Content = content }],
        MaxTokens = 3,
        Greedy = true,
        Device = "cpu",
    };

    private ModelSpec Spec() => new() { Requested = "minimax-m3-fixture", Modality = Modality.Text, LocalPath = _directory };

    /// <summary>The tiny weights plus a 64-token byte-level tokenizer (letters, digits, space, newline and one merge, the two control tokens last) and a minimal template.</summary>
    private void WriteCheckpoint(bool chatTemplate = true)
    {
        MiniMaxM3TinyCheckpoint.Write(_directory);
        Dictionary<string, int> vocab = [];
        foreach (char letter in "abcdefghijklmnopqrstuvwxyz") vocab[letter.ToString()] = vocab.Count;
        vocab["Ġ"] = vocab.Count;
        vocab["Ċ"] = vocab.Count;
        foreach (char digit in "0123456789") vocab[digit.ToString()] = vocab.Count;
        vocab["hi"] = vocab.Count;
        while (vocab.Count < VocabSize - 2) vocab[$"<filler{vocab.Count}>"] = vocab.Count;
        var tokenizer = new
        {
            version = "1.0",
            added_tokens = new[]
            {
                new { id = VocabSize - 2, content = "]~b]", special = true },
                new { id = VocabSize - 1, content = "[e~[", special = true },
            },
            model = new { type = "BPE", vocab, merges = new[] { "h i" } },
        };
        File.WriteAllText(Path.Combine(_directory, "tokenizer.json"), JsonSerializer.Serialize(tokenizer));
        if (chatTemplate)
        {
            File.WriteAllText(Path.Combine(_directory, "chat_template.jinja"),
                "{% for m in messages %}]~b]{{ m['role'] }}\n{{ m['content'] }}[e~[\n{% endfor %}{% if add_generation_prompt %}]~b]ai\n{% endif %}");
        }
    }

    [Fact]
    public async Task TextService_HostsTheDirectoryOnTheCpu_GeneratesGreedily_AndReusesTheLoadedSlot()
    {
        WriteCheckpoint();
        using InferenceEngine engine = new("cpu", 0);

        TextResult first = await engine.Text.GenerateAsync(Spec(), Request("hi"));
        TextResult second = await engine.Text.GenerateAsync(Spec(), Request("hi"));

        Assert.True(first.CompletionTokens is >= 1 and <= 3, $"completion tokens: {first.CompletionTokens}");
        Assert.True(first.PromptTokens > 0);
        Assert.Equal(first.Text, second.Text);
        Assert.Equal(first.CompletionTokens, second.CompletionTokens);
        Assert.Equal(1, engine.Text.CountTokens(Spec(), "hi"));
        Assert.True(engine.Text.Unload());
    }

    [Fact]
    public void ADirectoryWithoutAChatTemplate_IsRefusedByName()
    {
        WriteCheckpoint(chatTemplate: false);
        HfCheckpointInfo info = HfCheckpointDirectory.TryProbe(_directory)!;

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => HfTextDirectoryLoader.RequireSupported(info));

        Assert.Contains("chat_template.jinja", error.Message);
    }

    [Fact]
    public void TheReleaseConfig_IsRecognisedAsMiniMaxM3()
    {
        File.Copy(Path.Combine(AppContext.BaseDirectory, "Fixtures", "MiniMaxM3", "minimax_m3_1_flash_config.json"), Path.Combine(_directory, "config.json"));
        File.WriteAllBytes(Path.Combine(_directory, "model.safetensors"), []);
        File.WriteAllText(Path.Combine(_directory, "tokenizer.json"), "{}");
        File.WriteAllText(Path.Combine(_directory, "chat_template.jinja"), "");

        HfCheckpointInfo info = HfCheckpointDirectory.TryProbe(_directory)!;

        Assert.Equal("minimax_m3_vl", info.ModelType);
        HfTextDirectoryLoader.RequireSupported(info);
    }
}
