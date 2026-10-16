using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.ModelAssets.Gguf;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>A GGUF that sets <c>add_bos_token</c> starts every chat prompt with exactly one BOS, as llama.cpp does: it strips a BOS the
/// template rendered first, then its tokenizer adds one, whatever the template source says about <c>bos_token</c>.</summary>
public sealed class JinjaChatTemplateBosTests
{
    private static readonly ChatMessage[] Hello = [new ChatMessage("user", "hi")];

    private static readonly int Bos = LiteralControlTokenizer.IdOf(LiteralControlTokenizer.Bos);

    private static int[] Encode(string template, bool addBos)
    {
        GgufMetadata meta = new();
        meta.Add("tokenizer.chat_template", template);
        meta.Add("tokenizer.ggml.add_bos_token", addBos);
        LiteralControlTokenizer tokenizer = new();
        return GgufLanguageModel.BuildTemplate(meta, tokenizer).Encode(tokenizer, Hello, addGenerationPrompt: false);
    }

    [Theory]
    [InlineData("{{ messages[0]['content'] }}")]
    [InlineData("{{ bos_token }}{{ messages[0]['content'] }}")]
    [InlineData("{% if messages | length > 1 %}{{ bos_token }}{% endif %}{{ messages[0]['content'] }}")]
    public void AddBos_StartsThePromptWithExactlyOneBos(string template)
    {
        Assert.Equal([Bos, 'h', 'i'], Encode(template, addBos: true));
    }

    [Fact]
    public void AddBos_AlsoStartsTheChatMlFallbackWithBos()
    {
        // A template that fails to render for this conversation falls back to ChatML, which must still honour add_bos_token.
        int[] ids = Encode("{{ messages[0]['content'] | no_such_filter }}", addBos: true);

        Assert.Equal(Bos, ids[0]);
        Assert.Equal(LiteralControlTokenizer.IdOf(LiteralControlTokenizer.ImStart), ids[1]);
    }

    [Theory]
    [InlineData("{{ messages[0]['content'] }}")]
    [InlineData("{{ bos_token }}{{ messages[0]['content'] }}")]
    public void WithoutAddBos_TheRenderIsEncodedAsIs(string template)
    {
        int[] want = template.StartsWith("{{ bos_token }}", StringComparison.Ordinal) ? [Bos, 'h', 'i'] : ['h', 'i'];

        Assert.Equal(want, Encode(template, addBos: false));
    }
}
