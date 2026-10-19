using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Looped;

/// <summary>The prompt the engine builds from the real Dory GGUF's own chat template (header only, no weights): its generation prompt prefills
/// <c>&lt;think&gt;</c> unless thinking is off, no BOS is added, and the parser therefore starts in reasoning. Skips when the GGUF is absent (set
/// DORY_GGUF_PATH to NVIDIA-Nemotron-Labs-Dory1.0-4B-Q5_K_M.gguf).</summary>
public sealed class DoryChatTemplateTests
{
    private static readonly IReadOnlyList<ChatMessage> Conversation =
    [
        ChatMessage.System("You are Swarmie, the official in-app helper for SwarmUI."),
        ChatMessage.User("Explain in three sentences why the sky is blue, then list two things that would change its color."),
    ];

    private readonly ITestOutputHelper _output;

    public DoryChatTemplateTests(ITestOutputHelper output) => _output = output;

    private (ILlmTokenizer Tokenizer, IChatTemplate Template)? Load()
    {
        string? path = Environment.GetEnvironmentVariable("DORY_GGUF_PATH");
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            _output.WriteLine("SKIPPED: set DORY_GGUF_PATH to the Dory Q5_K_M GGUF.");
            return null;
        }
        using GgufLoader loader = new();
        loader.Load(path);
        Assert.Equal("dory", loader.Metadata.GetString("general.architecture"));
        ILlmTokenizer tokenizer = GgufLanguageModel.BuildTokenizer(loader.Metadata);
        return (tokenizer, GgufLanguageModel.BuildTemplate(loader.Metadata, tokenizer));
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void DefaultPromptEndsInsideAThinkBlock()
    {
        if (Load() is not var (tokenizer, template)) return;
        JinjaChatTemplate jinja = Assert.IsType<JinjaChatTemplate>(template);
        int[] ids = template.Encode(tokenizer, Conversation, addGenerationPrompt: true, enableThinking: null);
        int imStart = tokenizer.SpecialId("<|im_start|>")!.Value;
        Assert.Equal(imStart, ids[0]);
        Assert.Equal(tokenizer.Encode("<|im_start|>system\n" + Conversation[0].Content + "<|im_end|>\n", addSpecial: true),
            ids[..TurnEnd(ids, tokenizer, 1)]);
        Assert.Equal(tokenizer.Encode("<|im_start|>assistant\n<think>\n", addSpecial: true), ids[Array.LastIndexOf(ids, imStart)..]);
        Assert.True(jinja.ResolveParserState(tokenizer, Conversation, enableThinking: null, tools: null).ReasoningOpen);
        _output.WriteLine($"{ids.Length} prompt ids; tail [{string.Join(", ", ids[Array.LastIndexOf(ids, imStart)..])}]");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ThinkingOffPrefillsAnEmptyThinkBlock()
    {
        if (Load() is not var (tokenizer, template)) return;
        JinjaChatTemplate jinja = Assert.IsType<JinjaChatTemplate>(template);
        int[] ids = template.Encode(tokenizer, Conversation, addGenerationPrompt: true, enableThinking: false);
        int imStart = tokenizer.SpecialId("<|im_start|>")!.Value;
        Assert.Equal(tokenizer.Encode("<|im_start|>assistant\n<think></think>", addSpecial: true), ids[Array.LastIndexOf(ids, imStart)..]);
        Assert.False(jinja.ResolveParserState(tokenizer, Conversation, enableThinking: false, tools: null).ReasoningOpen);
    }

    /// <summary>The index just past the <paramref name="turn"/>-th <c>&lt;|im_end|&gt;</c> and the newline after it.</summary>
    private static int TurnEnd(int[] ids, ILlmTokenizer tokenizer, int turn)
    {
        int imEnd = tokenizer.SpecialId("<|im_end|>")!.Value;
        int seen = 0;
        for (int i = 0; i < ids.Length; i++)
        {
            if (ids[i] == imEnd && ++seen == turn) return i + 1 + tokenizer.EncodeOrdinary("\n").Length;
        }
        throw new InvalidOperationException($"No turn {turn} in the prompt.");
    }
}
