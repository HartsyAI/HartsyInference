using HartsyInference.LLM.ChatTemplates;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>The Hermes tool prompt: what a tool-less template receives, and the ChatML fallback it must keep matching byte for byte.</summary>
public sealed class HermesToolPromptTests
{
    private static readonly ToolSpec GetTime = ToolSpec.FromJson(
        "{\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"description\":\"time\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}");

    [Fact]
    public void TemplateOffersToolsOnlyForTheWord()
    {
        Assert.True(HermesToolPrompt.TemplateOffersTools("{% if tools %}x{% endif %}"));
        Assert.False(HermesToolPrompt.TemplateOffersTools("{% set tools_in_user_message = true %}"));
        Assert.False(HermesToolPrompt.TemplateOffersTools("{% for m in messages %}{{ m.content }}{% endfor %}"));
    }

    [Fact]
    public void ToolsBlockGoesIntoTheSystemTurnAndTheUserAsksAreKept()
    {
        IReadOnlyList<ChatMessage> rewritten = HermesToolPrompt.RewriteForToolLessTemplate([new ChatMessage("user", "what time is it")], [GetTime], null, null);
        Assert.Equal(["system", "user"], rewritten.Select(m => m.Role));
        Assert.Contains("<tools>", rewritten[0].Content, StringComparison.Ordinal);
        Assert.Contains("\"get_time\"", rewritten[0].Content, StringComparison.Ordinal);
        Assert.Equal("what time is it", rewritten[1].Content);
    }

    [Fact]
    public void AssistantCallsAndConsecutiveResultsAreWrittenAsHermesText()
    {
        List<ChatMessage> conversation =
        [
            new ChatMessage("user", "time?"),
            new ChatMessage("assistant", "") { ToolCalls = [new ChatToolCall("c1", "get_time", "{}")] },
            new ChatMessage("tool", "{\"now\":\"14:05\"}") { ToolCallId = "c1", Name = "get_time" },
            new ChatMessage("tool", "{\"ok\":true}") { ToolCallId = "c2", Name = "other" },
        ];
        IReadOnlyList<ChatMessage> rewritten = HermesToolPrompt.RewriteForToolLessTemplate(conversation, [GetTime], null, null);
        Assert.Equal(["system", "user", "assistant", "user"], rewritten.Select(m => m.Role));
        Assert.Contains("<tool_call>\n{\"name\": \"get_time\", \"arguments\": {}}\n</tool_call>", rewritten[2].Content, StringComparison.Ordinal);
        Assert.Equal("<tool_response>\n{\"now\":\"14:05\"}\n</tool_response>\n<tool_response>\n{\"ok\":true}\n</tool_response>", rewritten[3].Content);
    }

    [Fact]
    public void ForcedToolDirectiveEndsTheSystemTurn()
    {
        IReadOnlyList<ChatMessage> rewritten = HermesToolPrompt.RewriteForToolLessTemplate([new ChatMessage("user", "hi")], [GetTime], "Be brief.", "get_time");
        Assert.StartsWith("Be brief.", rewritten[0].Content, StringComparison.Ordinal);
        Assert.EndsWith(HermesToolPrompt.ForceDirective("get_time"), rewritten[0].Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ChatMlFallbackRendersTheSameToolsBlockAsTheSharedHelper()
    {
        // ChatML encodes each turn's text ordinarily, so the tools block's own <tool_call> text is plain text, not a control id.
        LiteralControlTokenizer tok = new();
        int[] ids = new ChatMlTemplate().Encode(tok, [new ChatMessage("user", "hi")], addGenerationPrompt: true, enableThinking: null, tools: [GetTime]);
        int imStart = LiteralControlTokenizer.IdOf(LiteralControlTokenizer.ImStart);
        int imEnd = LiteralControlTokenizer.IdOf(LiteralControlTokenizer.ImEnd);
        List<int> expected = [imStart];
        expected.AddRange(tok.EncodeOrdinary("system\nYou are a helpful assistant." + HermesToolPrompt.ToolsBlock([GetTime])));
        expected.Add(imEnd);
        expected.AddRange(tok.EncodeOrdinary("\n"));
        expected.Add(imStart);
        expected.AddRange(tok.EncodeOrdinary("user\nhi"));
        expected.Add(imEnd);
        expected.AddRange(tok.EncodeOrdinary("\n"));
        expected.Add(imStart);
        expected.AddRange(tok.EncodeOrdinary("assistant\n"));
        Assert.Equal(expected, ids);
    }
}
