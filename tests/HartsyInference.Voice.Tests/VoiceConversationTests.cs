using HartsyInference.Engine.Requests;
using HartsyInference.Voice.Turns;
using Xunit;

namespace HartsyInference.Voice.Tests;

/// <summary>The history the model sees: system message first, trimmed oldest-first to the token budget, never dropping
/// the system message or the last user message and never starting on a reply or a tool result whose call is gone.</summary>
public sealed class VoiceConversationTests
{
    private static int Words(string text) => text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;

    [Fact]
    public void UnderTheBudgetEverythingIsSentSystemFirst()
    {
        VoiceConversation conversation = new("Be brief.");
        conversation.AddUser("hello there");
        conversation.AddAssistant("hi, how can I help");
        conversation.AddUser("what time is it");

        IReadOnlyList<TextMessage> request = conversation.ToRequest(Words, maxTokens: 1_000);

        Assert.Equal([TextRole.System, TextRole.User, TextRole.Assistant, TextRole.User], request.Select(m => m.Role));
        Assert.Equal("Be brief.", request[0].Content);
        Assert.Equal("what time is it", request[^1].Content);
    }

    [Fact]
    public void TrimmingDropsTheOldestTurnsAndKeepsTheSystemAndLastUserMessage()
    {
        VoiceConversation conversation = new("Be brief.");
        for (int turn = 0; turn < 10; turn++)
        {
            conversation.AddUser($"question number {turn} is here");
            conversation.AddAssistant($"answer number {turn} is right here now");
        }
        conversation.AddUser("final question");
        int overhead = VoiceConversation.MessageOverheadTokens;
        int budget = 60;

        IReadOnlyList<TextMessage> request = conversation.ToRequest(Words, budget);

        Assert.Equal(TextRole.System, request[0].Role);
        Assert.Equal(TextRole.User, request[1].Role);
        Assert.Equal("final question", request[^1].Content);
        Assert.True(request.Sum(m => Words(m.Content) + overhead) <= budget);
        Assert.True(request.Count < 22);
        Assert.Contains("question number 9", request[^3].Content, StringComparison.Ordinal);
        // Trimming is permanent: the next request starts from what was kept.
        Assert.Equal(request.Count - 1, conversation.History.Count);
    }

    [Fact]
    public void AToolResultNeverOutlivesItsCall()
    {
        VoiceConversation conversation = new("sys");
        conversation.AddUser("what is the weather in a very long winded way please");
        NativeToolCall call = new() { Id = "call_0", Name = "weather", Arguments = "{\"city\": \"Paris\"}" };
        conversation.AddAssistant("let me look", [call]);
        conversation.AddToolResult(call, "sunny and warm all day long with a light breeze");
        conversation.AddAssistant("it is sunny");
        conversation.AddUser("thanks");

        // 30 tokens fit once the first question and the call are gone, which would leave the call's result first.
        IReadOnlyList<TextMessage> request = conversation.ToRequest(Words, maxTokens: 30);

        Assert.Equal([TextRole.System, TextRole.User], request.Select(m => m.Role));
        Assert.Equal("thanks", request[^1].Content);
    }

    [Fact]
    public void TheLastUserMessageIsKeptEvenOverBudget()
    {
        VoiceConversation conversation = new("a system prompt that is long");
        conversation.AddUser("one two three four five six seven eight nine ten");

        IReadOnlyList<TextMessage> request = conversation.ToRequest(Words, maxTokens: 3);

        Assert.Equal([TextRole.System, TextRole.User], request.Select(m => m.Role));
    }
}
