using System.Text.Json;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>MiniMax-M3.1-Flash's <c>tokenizer.json</c> and <c>chat_template.jinja</c> against the ids transformers produces for them
/// (<c>tests/python-reference/dump_minimax_m3_text_reference.py</c>). Needs the release's tokenizer files in the directory named by <c>MINIMAX_M3_DIR</c>.</summary>
public sealed class MiniMaxM3TextAssetsTests(ITestOutputHelper output)
{
    private const string BosToken = "]~b]";
    private const string EosToken = "[e~[";

    private static string? Directory() => Environment.GetEnvironmentVariable("MINIMAX_M3_DIR") is { Length: > 0 } dir && File.Exists(Path.Combine(dir, "tokenizer.json")) ? dir : null;

    private static JsonDocument Reference() =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "MiniMaxM3", "minimax_m3_text_reference.json")));

    [Fact]
    [Trait("Category", "Integration")]
    public void Tokenizer_MatchesTransformersIds()
    {
        if (Directory() is not string dir) { output.WriteLine("SKIPPED: set MINIMAX_M3_DIR to a MiniMax-M3.1-Flash tokenizer directory."); return; }
        (ILlmTokenizer tokenizer, _) = HfChatAssets.Load(dir, [], BosToken, EosToken);
        using JsonDocument reference = Reference();
        JsonElement texts = reference.RootElement.GetProperty("texts"), ids = reference.RootElement.GetProperty("textIds");
        for (int i = 0; i < texts.GetArrayLength(); i++)
            Assert.Equal(ids[i].EnumerateArray().Select(static e => e.GetInt32()), tokenizer.Encode(texts[i].GetString()!, addSpecial: true));
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ChatTemplate_MatchesTransformersIds()
    {
        if (Directory() is not string dir) { output.WriteLine("SKIPPED: set MINIMAX_M3_DIR to a MiniMax-M3.1-Flash tokenizer directory."); return; }
        (ILlmTokenizer tokenizer, IChatTemplate template) = HfChatAssets.Load(dir, [], BosToken, EosToken);
        using JsonDocument reference = Reference();
        JsonElement conversations = reference.RootElement.GetProperty("conversations"), ids = reference.RootElement.GetProperty("conversationIds");
        for (int i = 0; i < conversations.GetArrayLength(); i++)
        {
            ChatMessage[] messages = [.. conversations[i].EnumerateArray().Select(static m => new ChatMessage(m.GetProperty("role").GetString()!, m.GetProperty("content").GetString()!))];
            Assert.Equal(ids[i].EnumerateArray().Select(static e => e.GetInt32()), template.Encode(tokenizer, messages, addGenerationPrompt: true));
        }
    }
}
