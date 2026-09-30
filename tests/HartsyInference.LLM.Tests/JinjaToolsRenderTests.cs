using System.Text;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Tool plumbing through the chat templates: the Qwen2.5 <c>chat_template</c> (verbatim from the GGUF metadata) renders <c>tools</c>, assistant <c>tool_calls</c> and tool results from the context the engine builds, and the ChatML fallback renders the same conversation byte for byte.</summary>
public sealed class JinjaToolsRenderTests
{
    /// <summary>tokenizer.chat_template of Qwen2.5-1.5B-Instruct (the official Q8_0 GGUF), unchanged apart from the trailing newline. Some third-party GGUF copies double the braces in the <c>{"name": …}</c> literal; Jinja renders either literally.</summary>
    private const string Qwen25Template = """
        {%- if tools %}
            {{- '<|im_start|>system\n' }}
            {%- if messages[0]['role'] == 'system' %}
                {{- messages[0]['content'] }}
            {%- else %}
                {{- 'You are a helpful assistant.' }}
            {%- endif %}
            {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
            {%- for tool in tools %}
                {{- "\n" }}
                {{- tool | tojson }}
            {%- endfor %}
            {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
        {%- else %}
            {%- if messages[0]['role'] == 'system' %}
                {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
            {%- else %}
                {{- '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}
            {%- endif %}
        {%- endif %}
        {%- for message in messages %}
            {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
                {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
            {%- elif message.role == "assistant" %}
                {{- '<|im_start|>' + message.role }}
                {%- if message.content %}
                    {{- '\n' + message.content }}
                {%- endif %}
                {%- for tool_call in message.tool_calls %}
                    {%- if tool_call.function is defined %}
                        {%- set tool_call = tool_call.function %}
                    {%- endif %}
                    {{- '\n<tool_call>\n{"name": "' }}
                    {{- tool_call.name }}
                    {{- '", "arguments": ' }}
                    {{- tool_call.arguments | tojson }}
                    {{- '}\n</tool_call>' }}
                {%- endfor %}
                {{- '<|im_end|>\n' }}
            {%- elif message.role == "tool" %}
                {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
                    {{- '<|im_start|>user' }}
                {%- endif %}
                {{- '\n<tool_response>\n' }}
                {{- message.content }}
                {{- '\n</tool_response>' }}
                {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
                    {{- '<|im_end|>\n' }}
                {%- endif %}
            {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\n' }}
        {%- endif %}
        """;

    private const string WeatherTool = "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get the weather\","
        + "\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}";

    private const string Expected =
        "<|im_start|>system\nYou are a bot.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
        + "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
        + "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather\", "
        + "\"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}\n"
        + "</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        + "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n"
        + "<|im_start|>user\nWhat's the weather in Paris?<|im_end|>\n"
        + "<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call><|im_end|>\n"
        + "<|im_start|>user\n<tool_response>\n{\"temp\": 21}\n</tool_response><|im_end|>\n"
        + "<|im_start|>user\nthanks<|im_end|>\n"
        + "<|im_start|>assistant\n";

    private static List<ChatMessage> Conversation() =>
    [
        ChatMessage.System("You are a bot."),
        ChatMessage.User("What's the weather in Paris?"),
        ChatMessage.Assistant("") with { ToolCalls = [new ChatToolCall("call_1", "get_weather", "{\"city\": \"Paris\"}")] },
        ChatMessage.Tool("call_1", "{\"temp\": 21}") with { Name = "get_weather" },
        ChatMessage.User("thanks"),
    ];

    [Fact]
    public void Qwen25TemplateRendersToolsToolCallsAndToolResponses()
    {
        CaptureTokenizer tok = new();
        new JinjaChatTemplate(Qwen25Template).Encode(tok, Conversation(), addGenerationPrompt: true, enableThinking: null, [ToolSpec.FromJson(WeatherTool)]);
        Assert.Equal(Expected, tok.Rendered);
    }

    [Fact]
    public void Qwen25TemplateWithoutToolsGetsNullNotAnEmptyList()
    {
        CaptureTokenizer tok = new();
        new JinjaChatTemplate(Qwen25Template).Encode(tok, [ChatMessage.User("hi")], addGenerationPrompt: true);
        Assert.Equal("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", tok.Rendered);
        new JinjaChatTemplate(Qwen25Template).Encode(tok, [ChatMessage.User("hi")], addGenerationPrompt: true, enableThinking: null, []);
        Assert.DoesNotContain("# Tools", tok.Rendered);
    }

    [Fact]
    public void MessageContextCarriesToolCallIdNameAndStringArguments()
    {
        CaptureTokenizer tok = new();
        const string template = "{% for m in messages %}[{{ m.role }}|{{ m.name }}|{{ m.tool_call_id }}|"
            + "{% for c in m.tool_calls %}{{ c.id }}:{{ c.function.name }}:{{ 'str' if c.function.arguments is string else 'obj' }}{% endfor %}]{% endfor %}";
        List<ChatMessage> messages =
        [
            ChatMessage.Assistant("") with { ToolCalls = [new ChatToolCall("c1", "f", "not json"), new ChatToolCall("c2", "g", "{\"a\":1}")] },
            ChatMessage.Tool("c1", "r") with { Name = "f" },
        ];
        new JinjaChatTemplate(template).Encode(tok, messages, addGenerationPrompt: false);
        Assert.Equal("[assistant|||c1:f:strc2:g:obj][tool|f|c1|]", tok.Rendered);
    }

    [Fact]
    public void ChatMlFallbackRendersTheSameConversationByteForByte()
    {
        CaptureTokenizer tok = new();
        int[] ids = new ChatMlTemplate().Encode(tok, Conversation(), addGenerationPrompt: true, enableThinking: null, [ToolSpec.FromJson(WeatherTool)]);
        Assert.Equal(Expected, tok.Text(ids));
    }

    [Fact]
    public void ChatMlFallbackWithoutToolsIsUnchanged()
    {
        CaptureTokenizer tok = new();
        int[] ids = new ChatMlTemplate().Encode(tok, [ChatMessage.System("S"), ChatMessage.User("hi")], addGenerationPrompt: true);
        Assert.Equal("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n", tok.Text(ids));
    }

    [Fact]
    public void ChatMlFallbackWithoutSystemMessageUsesQwenDefaultForTheToolsBlock()
    {
        CaptureTokenizer tok = new();
        int[] ids = new ChatMlTemplate().Encode(tok, [ChatMessage.User("hi")], addGenerationPrompt: false, enableThinking: null, [ToolSpec.FromJson(WeatherTool)]);
        Assert.StartsWith("<|im_start|>system\nYou are a helpful assistant.\n\n# Tools\n", tok.Text(ids));
        Assert.EndsWith("</tool_call><|im_end|>\n<|im_start|>user\nhi<|im_end|>\n", tok.Text(ids));
    }

    /// <summary>Captures the rendered prompt: the Jinja path hands the whole string to <see cref="Encode"/>, the ChatML path hands pieces to <see cref="EncodeOrdinary"/> between control ids.</summary>
    private sealed class CaptureTokenizer : ILlmTokenizer
    {
        private const int ImStart = 0;
        private const int ImEnd = 1;
        private readonly List<string> _pieces = [];

        public string Rendered { get; private set; } = "";

        public int[] Encode(string text, bool addSpecial)
        {
            Rendered = text;
            return [];
        }

        public int[] EncodeOrdinary(string text)
        {
            _pieces.Add(text);
            return [_pieces.Count + 1];
        }

        public string Text(int[] ids)
        {
            StringBuilder sb = new();
            foreach (int id in ids) sb.Append(id switch { ImStart => "<|im_start|>", ImEnd => "<|im_end|>", _ => _pieces[id - 2] });
            return sb.ToString();
        }

        public string Decode(IReadOnlyList<int> ids) => "";

        public int? SpecialId(string token) => token switch { "<|im_start|>" => ImStart, "<|im_end|>" => ImEnd, _ => null };

        public int? BosId => null;

        public int? EosId => ImEnd;

        public IReadOnlyList<int> StopIds => [ImEnd];

        public string? BosToken => null;

        public string? EosToken => "<|im_end|>";
    }
}
