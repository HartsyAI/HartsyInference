using HartsyInference.LLM.ChatTemplates;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The reasoning effort a request sets reaches the DeepSeek-V4.1 thinking prompt: a different effort renders a different prompt, the default is 75, and an
/// effort outside [1, 100] is refused.</summary>
public sealed class DeepSeekV41ReasoningEffortTests
{
    private static string Render(int? effort) => HartsyInference.LLM.ChatTemplates.DeepSeekV41Encoder.RenderText(
        [ChatMessage.User("Solve it.")],
        new EncodeOptions { Thinking = true, ReasoningEffort = effort });

    [Fact]
    public void A_Different_Effort_Renders_A_Different_Thinking_Prompt()
    {
        Assert.NotEqual(Render(100), Render(75));
        // null is the template's default, which is 75
        Assert.Equal(Render(75), Render(null));
    }

    [Fact]
    public void An_Effort_Outside_1_To_100_Is_Refused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Render(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Render(101));
    }
}
