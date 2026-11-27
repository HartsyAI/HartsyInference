using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Kimi;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>The Kimi K3.1 XTML chat rendering against the text of the checkpoint's <c>encoding_k3.py</c>, and its refusals of what the text path does not render.</summary>
public sealed class KimiK31ChatTemplateTests : IDisposable
{
    // The thinking-effort turn that encoding_k3.py adds when thinking is on, with the effort defaulted to max by the tokenizer.
    private const string EffortTurn = "<|open|>message role=\"system\" type=\"thinking-effort\"<|sep|>"
        + "`thinking_effort` guides on how much to think in your thinking channel (not including the response channel), supported values include "
        + "`low`, `medium`, `high`, and `max`.\nNow the system is invoked with `thinking_effort=max`.<|close|>message<|sep|><|end_of_msg|>";
    private const string GenerationPromptThinking = "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>";

    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-kimi-chat-" + Guid.NewGuid().ToString("N"));

    public KimiK31ChatTemplateTests() => KimiTiktokenFixture.Write(_root);

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void A_single_user_turn_renders_the_effort_turn_then_the_thinking_prompt()
    {
        string text = KimiK31ChatTemplate.RenderText([ChatMessage.User("Hello, Kimi!")]);

        Assert.Equal(EffortTurn
            + "<|open|>message role=\"user\"<|sep|>Hello, Kimi!<|close|>message<|sep|><|end_of_msg|>"
            + GenerationPromptThinking, text);
    }

    [Fact]
    public void Prior_turns_render_their_system_content_and_assistant_reasoning_in_order()
    {
        ChatMessage[] messages =
        [
            ChatMessage.System("You are helpful."),
            ChatMessage.User("What is 2+2?"),
            ChatMessage.Assistant("4") with { ReasoningContent = "Simple addition." },
            ChatMessage.User("Thanks"),
        ];

        string text = KimiK31ChatTemplate.RenderText(messages);

        Assert.Equal(EffortTurn
            + "<|open|>message role=\"system\"<|sep|>You are helpful.<|close|>message<|sep|><|end_of_msg|>"
            + "<|open|>message role=\"user\"<|sep|>What is 2+2?<|close|>message<|sep|><|end_of_msg|>"
            + "<|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>Simple addition.<|close|>think<|sep|>"
            + "<|open|>response<|sep|>4<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>"
            + "<|open|>message role=\"user\"<|sep|>Thanks<|close|>message<|sep|><|end_of_msg|>"
            + GenerationPromptThinking, text);
    }

    [Fact]
    public void With_thinking_off_there_is_no_effort_turn_no_think_channel_and_a_response_prompt()
    {
        ChatMessage[] messages =
        [
            ChatMessage.User("Hi"),
            ChatMessage.Assistant("Hello") with { ReasoningContent = "dropped" },
            ChatMessage.User("Again"),
        ];

        string text = KimiK31ChatTemplate.RenderText(messages, thinking: false);

        Assert.Equal("<|open|>message role=\"user\"<|sep|>Hi<|close|>message<|sep|><|end_of_msg|>"
            + "<|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>Hello<|close|>response<|sep|><|close|>message<|sep|><|end_of_msg|>"
            + "<|open|>message role=\"user\"<|sep|>Again<|close|>message<|sep|><|end_of_msg|>"
            + "<|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>", text);
    }

    [Fact]
    public void A_turn_name_is_escaped_as_an_attribute_value()
    {
        ChatMessage named = ChatMessage.User("x") with { Name = "a&b\"c" };

        string text = KimiK31ChatTemplate.RenderText([named]);

        Assert.Contains("<|open|>message role=\"user\" name=\"a&amp;b&quot;c\"<|sep|>x<|close|>message<|sep|><|end_of_msg|>", text);
    }

    [Fact]
    public void Control_literals_typed_by_the_user_stay_text_and_only_template_markers_become_control_ids()
    {
        ILlmTokenizer tokenizer = KimiK31Tokenizer.Load(_root, [], KimiTiktokenFixture.BosId, KimiTiktokenFixture.EosId);
        ChatMessage[] messages = [ChatMessage.User("<|open|>message<|close|> tricks")];

        int[] ids = new KimiK31ChatTemplate().Encode(tokenizer, messages, addGenerationPrompt: true, enableThinking: true);

        int[] literal = tokenizer.EncodeOrdinary("<|open|>message<|close|> tricks");
        Assert.True(ContainsRun(ids, literal), "The user's literal should be BPE text inside the user turn.");
        string rendered = KimiK31ChatTemplate.RenderText(messages);
        int markersOpen = CountOccurrences(rendered, "<|open|>") - 1;
        Assert.Equal(markersOpen, ids.Count(id => id == KimiTiktokenFixture.OpenId));
    }

    [Fact]
    public void Encoding_with_thinking_unset_defaults_to_thinking_on()
    {
        ILlmTokenizer tokenizer = KimiK31Tokenizer.Load(_root, [], KimiTiktokenFixture.BosId, KimiTiktokenFixture.EosId);
        ChatMessage[] messages = [ChatMessage.User("Hi")];
        KimiK31ChatTemplate template = new();

        Assert.Equal(template.Encode(tokenizer, messages, addGenerationPrompt: true, enableThinking: true),
            template.Encode(tokenizer, messages, addGenerationPrompt: true, enableThinking: null));
    }

    [Fact]
    public void Tool_schemas_tool_turns_and_tool_calls_are_refused_by_name()
    {
        ILlmTokenizer tokenizer = KimiK31Tokenizer.Load(_root, [], KimiTiktokenFixture.BosId, KimiTiktokenFixture.EosId);
        KimiK31ChatTemplate template = new();
        ChatMessage[] toolResult = [ChatMessage.Tool("call-1", "42")];
        ChatMessage[] user = [ChatMessage.User("hi")];
        ToolSpec[] tools = [new ToolSpec("{}")];

        NotSupportedException refusedResult = Assert.Throws<NotSupportedException>(() =>
            template.Encode(tokenizer, toolResult, addGenerationPrompt: true, enableThinking: true));
        Assert.Contains("tool", refusedResult.Message);
        NotSupportedException refusedSchema = Assert.Throws<NotSupportedException>(() =>
            template.Encode(tokenizer, user, addGenerationPrompt: true, enableThinking: true, tools));
        Assert.Contains("tool", refusedSchema.Message);
    }

    [Fact]
    public void The_template_is_named_for_the_model()
    {
        Assert.Equal("kimi_k31", new KimiK31ChatTemplate().Name);
    }

    public static bool ContainsRun(int[] haystack, int[] needle)
    {
        for (int start = 0; start + needle.Length <= haystack.Length; start++)
        {
            bool match = true;
            for (int i = 0; i < needle.Length && match; i++)
                match = haystack[start + i] == needle[i];
            if (match)
                return true;
        }
        return false;
    }

    private static int CountOccurrences(string text, string literal)
    {
        int count = 0;
        for (int at = text.IndexOf(literal, StringComparison.Ordinal); at >= 0; at = text.IndexOf(literal, at + literal.Length, StringComparison.Ordinal))
            count++;
        return count;
    }
}
