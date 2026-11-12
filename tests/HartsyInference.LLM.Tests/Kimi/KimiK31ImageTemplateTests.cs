using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Kimi;
using HartsyInference.LLM.Kimi.Vision;
using Xunit;

namespace HartsyInference.LLM.Tests.Kimi;

/// <summary>The XTML template's image parts: the processor's media placeholder for the image's size, before the text, and a refusal for an image whose
/// size is unknown.</summary>
public sealed class KimiK31ImageTemplateTests
{
    [Fact]
    public void Image_block_renders_the_processor_placeholder_before_its_text()
    {
        ChatMessage user = new("user", "What is this?")
        {
            Blocks = [new ImageBlock(0) { Width = 28, Height = 20 }, new TextBlock("What is this?")],
        };

        string text = KimiK31ChatTemplate.RenderText([user], thinking: false, addGenerationPrompt: false);

        string placeholder = "<|media_begin|>image 28x20<|media_content|><|media_pad|><|media_end|>";
        int at = text.IndexOf(placeholder, StringComparison.Ordinal);
        Assert.True(at >= 0, $"no placeholder in: {text}");
        Assert.Equal(at + placeholder.Length, text.IndexOf("What is this?", StringComparison.Ordinal));
        Assert.Equal(placeholder, KimiVisionInput.MediaPrompt(28, 20));
    }

    [Fact]
    public void Image_block_without_a_size_is_refused_by_name()
    {
        ChatMessage user = new("user", "hi") { Blocks = [new ImageBlock(0), new TextBlock("hi")] };

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() => KimiK31ChatTemplate.RenderText([user], thinking: false));

        Assert.Contains("pixel size", refused.Message, StringComparison.Ordinal);
    }
}
