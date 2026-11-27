using HartsyInference.LLM.ChatTemplates;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Tests.ChatTemplates;

/// <summary>A turn with an image block: the Jinja templates keep the plain content string unless they opt in, so a checkpoint that does not read
/// content lists (Kimi K3.1 sends such turns) renders exactly as it did before the image block was added.</summary>
public sealed class JinjaImageBlockTests
{
    private const string Source = "{% for m in messages %}[{{ m.role }}:{{ m.content }}]{% endfor %}{% if add_generation_prompt %}A{% endif %}";

    [Fact]
    public void PlainTemplate_RendersAnImageTurnAsItsContentString()
    {
        JinjaChatTemplate template = new(Source);
        ChatMessage plain = new("user", "look");
        ChatMessage withImage = new("user", "look") { Blocks = [new TextBlock("look"), new ImageBlock(0)] };
        FakeTokenizer tokenizer = new();

        Assert.Equal(template.Encode(tokenizer, [plain], true), template.Encode(tokenizer, [withImage], true));
        Assert.Equal("[user:look]A", Render(tokenizer, template, withImage));
    }

    [Fact]
    public void OptedInTemplate_RendersAnImageTurnAsAContentList()
    {
        // The opt-in template reads the list: a text item's text, an image item's type.
        const string listSource = "{% for m in messages %}[{{ m.role }}:{% for i in m.content %}{% if i.type == 'image' %}<img>{% else %}{{ i.text }}{% endif %}{% endfor %}]{% endfor %}";
        JinjaChatTemplate template = new(listSource, renderImageBlocks: true);
        ChatMessage withImage = new("user", "look") { Blocks = [new TextBlock("look"), new ImageBlock(0)] };
        Assert.Equal("[user:look<img>]", Render(new FakeTokenizer(), template, withImage));
    }

    private static string Render(ILlmTokenizer tokenizer, JinjaChatTemplate template, ChatMessage message) =>
        tokenizer.Decode(template.Encode(tokenizer, [message], true));

    private sealed class FakeTokenizer : ILlmTokenizer
    {
        public int[] Encode(string text, bool addSpecial) => EncodeOrdinary(text);
        public int[] EncodeOrdinary(string text) => text.Select(c => (int)c).ToArray();
        public string Decode(IReadOnlyList<int> ids) => string.Concat(ids.Select(i => (char)i));
        public int? SpecialId(string token) => null;
        public int? BosId => null;
        public int? EosId => null;
        public IReadOnlyList<int> StopIds => Array.Empty<int>();
        public string? BosToken => null;
        public string? EosToken => null;
    }
}
