using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Multimodal;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>The turn tail the Qwen3.5-VL prompt ends with is rendered by the loaded model's own template, so each template decides its think prefix.
/// Two templates are pinned: the SenseNova-Loop-9B template, and the Qwen3.5-9B template (unsloth, HF tokenizer_config, which opens <c>&lt;think&gt;\n</c>
/// when enable_thinking is unset and prefills an empty block only when it is false).</summary>
public sealed class Qwen35VlAssistantTurnTests
{
    private const string Header = "<|im_end|>\n<|im_start|>assistant\n";

    public static IEnumerable<object?[]> Cases()
    {
        foreach (string template in new[] { "SenseNovaLoop", "Qwen35Vl9B" })
        {
            yield return [template, "unset", Header + "<think>\n", true];
            yield return [template, "on", Header + "<think>\n", true];
            yield return [template, "off", Header + "<think>\n\n</think>\n\n", false];
        }
    }

    private static bool? Thinking(string mode) => mode switch { "on" => true, "off" => false, _ => null };

    [Theory]
    [MemberData(nameof(Cases))]
    public void TheLoadedTemplateRendersTheTailAndItsReasoningState(string template, string mode, string wantTail, bool wantOpen)
    {
        string source = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", template, "chat_template.jinja"));
        JinjaChatTemplate jinja = new(source);
        string tail = jinja.TurnTailAfterUser(new FakeTokenizer(), Thinking(mode));
        Assert.Equal(wantTail, tail);
        Assert.Equal(wantOpen, Qwen35VlGenerator.ReasoningOpen(tail));
    }

    /// <summary>Only the tokens the template chooses matter here; the tests exercise the turn tail, not encoding.</summary>
    private sealed class FakeTokenizer : ILlmTokenizer
    {
        private static readonly string[] Specials = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|vision_start|>", "<|vision_end|>"];

        public int[] Encode(string text, bool addSpecial) => EncodeOrdinary(text);
        public int[] EncodeOrdinary(string text) => text.Select(c => (int)c).ToArray();
        public string Decode(IReadOnlyList<int> ids) => string.Concat(ids.Select(i => (char)i));
        public int? SpecialId(string token) => Array.IndexOf(Specials, token) is var at and >= 0 ? 1000 + at : null;
        public int? BosId => null;
        public int? EosId => null;
        public IReadOnlyList<int> StopIds => Array.Empty<int>();
        public string? BosToken => null;
        public string? EosToken => null;
    }
}
