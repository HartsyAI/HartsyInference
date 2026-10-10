using System.Text.Json;
using HartsyInference.API.Endpoints;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>The timings the chat route reports in the llama.cpp names, and the thinking switch llama.cpp-style clients send.
/// These are the fields Strata's benchmark harness (bench/bench_vs_llama.py) reads from every reply.</summary>
public sealed class CompatTimingsTests
{
    [Fact]
    public void Timings_ReportTokensPerSecondFromPrefillAndDecodeWallTime()
    {
        ChatTimings t = ChatTimings.From(promptTokens: 4000, prefillMs: 2000, completionTokens: 256, decodeMs: 4000);

        Assert.Equal(4000, t.PromptN);
        Assert.Equal(2000.0, t.PromptPerSecond, precision: 6);
        Assert.Equal(256, t.PredictedN);
        Assert.Equal(64.0, t.PredictedPerSecond, precision: 6);
    }

    [Fact]
    public void Timings_ReportZeroSpeedForAZeroDurationNotInfinity()
    {
        ChatTimings t = ChatTimings.From(promptTokens: 10, prefillMs: 0, completionTokens: 5, decodeMs: 0);

        Assert.Equal(0.0, t.PromptPerSecond);
        Assert.Equal(0.0, t.PredictedPerSecond);
        Assert.True(double.IsFinite(t.PromptPerSecond) && double.IsFinite(t.PredictedPerSecond));
    }

    [Fact]
    public void Timings_SerializeWithTheLlamaCppPropertyNames()
    {
        string json = JsonSerializer.Serialize(ChatTimings.From(9, 12.5, 8, 80));

        foreach (string name in new[] { "prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second" })
            Assert.Contains($"\"{name}\":", json);
    }

    [Fact]
    public void ChatResponse_CarriesTimingsWhenMeasuredAndOmitsThemOtherwise()
    {
        ChatCompletionResponse measured = Response(ChatTimings.From(9, 12.5, 8, 80));
        ChatCompletionResponse unmeasured = Response(timings: null);

        Assert.Contains("\"timings\":{\"prompt_n\":9,", JsonSerializer.Serialize(measured));
        Assert.DoesNotContain("\"timings\"", JsonSerializer.Serialize(unmeasured));
    }

    [Theory]
    [InlineData("enabled", null, true)]
    [InlineData("disabled", null, false)]
    [InlineData(null, false, false)]
    [InlineData(null, true, true)]
    [InlineData("enabled", false, true)]
    [InlineData("disabled", true, false)]
    [InlineData(null, null, null)]
    public void ResolveThinking_PrefersTheThinkingFieldThenTheChatTemplateKwargs(string? thinkingType, bool? kwargsEnableThinking, bool? expected)
    {
        ChatThinkingDto? thinking = thinkingType is null ? null : new ChatThinkingDto { Type = thinkingType };
        ChatTemplateKwargsDto? kwargs = kwargsEnableThinking is null ? null : new ChatTemplateKwargsDto { EnableThinking = kwargsEnableThinking };

        Assert.Equal(expected, CompatEndpoints.ResolveThinking(thinking, kwargs));
    }

    [Fact]
    public void ChatRequest_ReadsChatTemplateKwargsEnableThinking()
    {
        ChatCompletionRequest req = JsonSerializer.Deserialize<ChatCompletionRequest>(
            "{\"model\":\"m\",\"messages\":[],\"chat_template_kwargs\":{\"enable_thinking\":false},\"cache_prompt\":false,\"top_k\":20}")!;

        Assert.False(req.ChatTemplateKwargs!.EnableThinking);
        Assert.Equal(20, req.TopK);
    }

    private static ChatCompletionResponse Response(ChatTimings? timings) => new()
    {
        Id = "chatcmpl-test",
        Created = 1,
        Model = "m",
        Choices = [new ChatCompletionChoice { Message = new ChatMessageDto { Role = "assistant", Content = "hi" }, FinishReason = "stop" }],
        Usage = new ChatUsage { PromptTokens = 9, CompletionTokens = 8 },
        Timings = timings,
    };
}
