using HartsyInference.LLM.ChatTemplates;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Unit tests for the Jinja chat-template engine and the byte-level codec (the Phase-1 per-model
/// prompting machinery).</summary>
public sealed class JinjaAndTokenizerTests
{
    private static Dictionary<string, object?> Ctx(bool addGen, params (string Role, string Content)[] messages)
    {
        List<object?> msgs = [];
        foreach ((string r, string c) in messages)
            msgs.Add(new Dictionary<string, object?> { ["role"] = r, ["content"] = c });
        return new Dictionary<string, object?>
        {
            ["messages"] = msgs,
            ["add_generation_prompt"] = addGen,
            ["bos_token"] = "<s>",
            ["eos_token"] = "</s>",
        };
    }

    [Fact]
    public void Jinja_ChatMl_ForIfSetLoop()
    {
        const string tmpl =
            "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}" +
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
        JinjaEngine engine = new(tmpl);
        string outp = engine.Render(Ctx(true, ("system", "You are helpful."), ("user", "Hi")));
        Assert.Equal(
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
            outp);
    }

    [Fact]
    public void Jinja_Llama3Style_BosConcatTrimLoopLast()
    {
        // Exercises: bos_token, '+'/'~' concat, .strip(), loop.last, whitespace-control, if/else.
        const string tmpl =
            "{{- bos_token }}" +
            "{%- for m in messages %}" +
            "{{- '<|h|>' + m['role'] + '<|e|>\n\n' + m['content'] | trim }}" +
            "{%- if not loop.last %}{{- '<|eot|>' }}{%- endif %}" +
            "{%- endfor %}" +
            "{%- if add_generation_prompt %}{{- '<|h|>assistant<|e|>\n\n' }}{%- endif %}";
        JinjaEngine engine = new(tmpl);
        string outp = engine.Render(Ctx(true, ("user", "  Hello  ")));
        Assert.Equal("<s><|h|>user<|e|>\n\nHello<|h|>assistant<|e|>\n\n", outp);
    }

    [Fact]
    public void Jinja_IsDefined_And_Ternary()
    {
        JinjaEngine engine = new("{% if tools is defined and tools is not none %}T{% else %}{{ 'no' if missing is not defined else 'yes' }}{% endif %}");
        // tools/missing not in context → 'tools is defined' false → else → 'missing is not defined' true → 'no'.
        Assert.Equal("no", engine.Render(new Dictionary<string, object?>()));
    }

    [Fact]
    public void Jinja_StringLiteralContainingBraces_NotMistakenForCloseTag()
    {
        // Regression: Qwen2/2.5/3 tool-call templates embed '{{' and '}}' inside a quoted string literal.
        // The lexer must skip string literals when scanning for the block-closing '}}' / '%}', otherwise it
        // truncates the expression mid-string → "Unterminated string in Jinja expr". (JinjaEngine.cs Lexer)
        const string tmpl =
            "{{- \"<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call>\" }}";
        JinjaEngine engine = new(tmpl);
        string outp = engine.Render(new Dictionary<string, object?>());
        Assert.Equal("<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>", outp);
    }

    [Fact]
    public void Jinja_CommentsAndSlice()
    {
        JinjaEngine engine = new("{# header #}{% for m in messages[1:] %}{{ m['content'] }}{% endfor %}");
        string outp = engine.Render(Ctx(false, ("system", "S"), ("user", "A"), ("user", "B")));
        Assert.Equal("AB", outp); // messages[1:] drops the system message
    }

    [Fact]
    public void ByteLevelCodec_RoundTripsSpacesNewlinesUnicode()
    {
        foreach (string s in new[] { " Hello world", "line1\nline2\n\n", "café — π", "tabs\tand spaces  " })
            Assert.Equal(s, ByteLevelCodec.Decode(ByteLevelCodec.Encode(s)));
    }

    [Fact]
    public void ByteLevelCodec_DecodesGpt2Markers()
    {
        // 'Ġ' (U+0120) is space, 'Ċ' (U+010A) is newline in GPT-2 byte-level space.
        Assert.Equal(" the\n", ByteLevelCodec.Decode("ĠtheĊ"));
    }

    // ── JinjaChatTemplate: normalize-and-retry for strict / system-less templates ───────────────────────

    /// <summary>A Mistral-v0.3-style template: no system role, strict user/assistant alternation. Raises
    /// (via the template's own raise_exception) the moment a system message appears or two same-role turns
    /// are adjacent — exactly the conditions JinjaChatTemplate.Encode must normalize away.</summary>
    private const string MistralStyleTemplate =
        "{{- bos_token }}" +
        "{%- for message in messages %}" +
        "{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}" +
        "{{- raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}" +
        "{%- endif %}" +
        "{%- if message['role'] == 'user' %}{{- '[INST] ' + message['content'] + ' [/INST]' }}" +
        "{%- elif message['role'] == 'assistant' %}{{- ' ' + message['content'] + eos_token }}" +
        "{%- endif %}" +
        "{%- endfor %}";

    /// <summary>Captures the rendered string handed to the tokenizer so tests can assert on the prompt text.</summary>
    private sealed class CapturingTokenizer : ILlmTokenizer
    {
        public string Last = "";
        public int[] Encode(string text, bool addSpecial) { Last = text; return [text.Length]; }
        public int[] EncodeOrdinary(string text) => [text.Length];
        public string Decode(IReadOnlyList<int> ids) => "";
        public int? SpecialId(string token) => null;
        public int? BosId => null;
        public int? EosId => null;
        public IReadOnlyList<int> StopIds => [];
        public string? BosToken => "<s>";
        public string? EosToken => "</s>";
    }

    [Fact]
    public void JinjaChatTemplate_FoldsSystemIntoFirstUser_ForSystemlessTemplate()
    {
        JinjaChatTemplate template = new(MistralStyleTemplate);
        CapturingTokenizer tok = new();
        // [system, user] would make the template raise (system where user must be); Encode must fold + retry.
        template.Encode(tok, [ChatMessage.System("You are helpful."), ChatMessage.User("Hi")], addGenerationPrompt: true);
        Assert.Equal("<s>[INST] You are helpful.\n\nHi [/INST]", tok.Last);
    }

    [Fact]
    public void JinjaChatTemplate_MergesConsecutiveSameRole_ForStrictTemplate()
    {
        JinjaChatTemplate template = new(MistralStyleTemplate);
        CapturingTokenizer tok = new();
        // Two adjacent user turns (e.g. an orphaned user message from a failed prior turn) would break
        // strict alternation; Encode must merge them into one.
        template.Encode(tok, [ChatMessage.User("A"), ChatMessage.User("B")], addGenerationPrompt: true);
        Assert.Equal("<s>[INST] A\n\nB [/INST]", tok.Last);
    }

    [Fact]
    public void JinjaChatTemplate_LeavesValidConversationUntouched()
    {
        JinjaChatTemplate template = new(MistralStyleTemplate);
        CapturingTokenizer tok = new();
        // Already valid (user/assistant/user) — no raise, so the original render is used as-is.
        template.Encode(tok, [ChatMessage.User("A"), ChatMessage.Assistant("B"), ChatMessage.User("C")], addGenerationPrompt: false);
        Assert.Equal("<s>[INST] A [/INST] B</s>[INST] C [/INST]", tok.Last);
    }
}
