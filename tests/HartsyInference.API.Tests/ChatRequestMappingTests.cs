using System.Text.Json;
using HartsyInference.API;
using HartsyInference.API.Endpoints;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>The OpenAI request fields a chat call carries into the native request: <c>reasoning_effort</c> (a name the engine defines, or an integer), <c>thinking</c>,
/// <c>user</c> and <c>priority</c>. A value the engine does not define is refused, not guessed at.</summary>
public sealed class ChatRequestMappingTests
{
    private static readonly JsonSerializerOptions s_options = new(JsonSerializerDefaults.Web);

    /// <summary>A chat request with one user message and <paramref name="extra"/> appended to its object (a leading comma included).</summary>
    private static TextRequest Map(string extra)
    {
        ChatCompletionRequest req = JsonSerializer.Deserialize<ChatCompletionRequest>(
            $$"""{"model":"m","messages":[{"role":"user","content":"hi"}]{{extra}}}""", s_options)!;
        return CompatEndpoints.ToTextRequest(req);
    }

    [Theory]
    [InlineData(",\"reasoning_effort\":\"max\"", 100)]
    [InlineData(",\"reasoning_effort\":\"high\"", 75)]
    [InlineData(",\"reasoning_effort\":\"low\"", 50)]
    [InlineData(",\"reasoning_effort\":42", 42)]
    [InlineData(",\"reasoning_effort\":\"42\"", 42)]
    [InlineData("", null)]
    public void Reasoning_Effort_Maps_To_The_Engine_Scale(string extra, int? expected)
    {
        Assert.Equal(expected, Map(extra).ReasoningEffort);
    }

    [Theory]
    [InlineData(",\"reasoning_effort\":\"medium\"")]
    [InlineData(",\"reasoning_effort\":0")]
    [InlineData(",\"reasoning_effort\":101")]
    [InlineData(",\"reasoning_effort\":99999999999")]
    [InlineData(",\"reasoning_effort\":1.5")]
    [InlineData(",\"reasoning_effort\":true")]
    public void A_Reasoning_Effort_The_Engine_Does_Not_Define_Is_Refused(string extra)
    {
        Assert.Throws<ArgumentException>(() => Map(extra));
    }

    [Fact]
    public void Thinking_Maps_To_The_Existing_Enable_Thinking_Flag()
    {
        Assert.True(Map(",\"thinking\":{\"type\":\"enabled\"}").EnableThinking);
        Assert.False(Map(",\"thinking\":{\"type\":\"disabled\"}").EnableThinking);
        Assert.Null(Map("").EnableThinking);
        Assert.Throws<ArgumentException>(() => Map(",\"thinking\":{\"type\":\"maybe\"}"));
    }

    [Theory]
    [InlineData(",\"thinking\":{\"type\":\"Enabled\"}")]
    [InlineData(",\"priority\":\"High\"")]
    public void Thinking_And_Priority_Values_Are_Case_Sensitive(string extra)
    {
        Assert.Throws<ArgumentException>(() => Map(extra));
    }

    [Fact]
    public void User_Passes_Through_And_Priority_Maps_To_The_Queue_Priority()
    {
        TextRequest mapped = Map(",\"user\":\"alice\",\"priority\":\"high\"");

        Assert.Equal("alice", mapped.User);
        Assert.Equal(RequestPriority.High, mapped.Priority);
        Assert.Null(Map("").Priority);
        Assert.Throws<ArgumentException>(() => Map(",\"priority\":\"urgent\""));
    }
}
