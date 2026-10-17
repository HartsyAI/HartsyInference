using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Only the DeepSeek-V4.1 template takes a reasoning effort. Every other template renders the same prompt with an effort as without one, through the prompt
/// builder generation uses.</summary>
public sealed class ReasoningEffortOtherTemplatesTests
{
    /// <summary>A Qwen3-style thinking template that would also print an effort, were one ever handed to Jinja.</summary>
    private const string ThinkingTemplate = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
        + "{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
        + "{% if reasoning_effort is defined %}effort={{ reasoning_effort }}{% endif %}";

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ChatMl_And_Jinja_Templates_Render_The_Same_Prompt_With_And_Without_An_Effort(bool thinking)
    {
        CharTokenizer tokenizer = new();
        foreach (IChatTemplate template in new IChatTemplate[] { new ChatMlTemplate(), new JinjaChatTemplate(ThinkingTemplate) })
        {
            int[] without = PromptBuilder.BuildPromptIds(Request(thinking, effort: null), tokenizer, template);
            int[] with = PromptBuilder.BuildPromptIds(Request(thinking, effort: 100), tokenizer, template);

            Assert.Equal(without, with);
        }
        // The Jinja template rendered itself rather than falling back to ChatML.
        string rendered = CharTokenizer.Text(PromptBuilder.BuildPromptIds(Request(thinking, effort: 100), tokenizer, new JinjaChatTemplate(ThinkingTemplate)));
        Assert.Equal("<|im_start|>user\nSolve it.<|im_end|>\n<|im_start|>assistant\n" + (thinking ? "" : "<think>\n\n</think>\n\n"), rendered);
    }

    private static GenerationRequest Request(bool thinking, int? effort) => new()
    {
        Messages = [ChatMessage.User("Solve it.")],
        EnableThinking = thinking,
        ReasoningEffort = effort,
    };
}
