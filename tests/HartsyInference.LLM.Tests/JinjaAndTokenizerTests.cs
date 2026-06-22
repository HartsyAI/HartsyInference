using HartsyInference.LLM.ChatTemplates;
using HartsyInference.Tokenizers;
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
}
