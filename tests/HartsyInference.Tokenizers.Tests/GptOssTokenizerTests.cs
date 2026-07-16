using System.Text.Json;
using HartsyInference.Tokenizers;
using Xunit;

namespace HartsyInference.Tokenizers.Tests;

/// <summary>GPT-OSS (o200k_harmony) tokenizer tests. The parity test compares against a token dump
/// produced by the upstream HuggingFace <c>tokenizers</c> library on the tokenizer.json embedded in
/// the Lens GPT-OSS encoder checkpoint — set <c>GPT_OSS_VOCAB_DIR</c> (dir holding
/// <c>gpt_oss_vocab.json</c> + <c>gpt_oss_merges.txt</c>) and <c>GPT_OSS_ORACLE_JSON</c> to run it.</summary>
public sealed class GptOssTokenizerTests
{
    private static GptOssTokenizer? TryCreate()
    {
        string? dir = Environment.GetEnvironmentVariable("GPT_OSS_VOCAB_DIR");
        if (string.IsNullOrEmpty(dir)) return null;
        string vocab = Path.Combine(dir, "gpt_oss_vocab.json");
        string merges = Path.Combine(dir, "gpt_oss_merges.txt");
        if (!File.Exists(vocab) || !File.Exists(merges)) return null;
        return new GptOssTokenizer(vocab, merges);
    }

    [Fact]
    public void RenderChatTemplate_Contains_Harmony_Preamble_And_Final_Channel()
    {
        string rendered = GptOssTokenizer.RenderChatTemplate("a corgi");
        Assert.Contains("You are ChatGPT, a large language model trained by OpenAI.", rendered);
        Assert.Contains("Knowledge cutoff: 2024-06", rendered);
        Assert.Contains($"Current date: {GptOssTokenizer.ChatTemplateDate}", rendered);
        Assert.Contains("<|start|>developer<|message|># Instructions", rendered);
        Assert.Contains($"<|start|>user<|message|>a corgi<|end|>", rendered);
        Assert.Contains("<|start|>assistant<|channel|>analysis<|message|>", rendered);
        Assert.EndsWith("<|start|>assistant<|channel|>final<|message|>", rendered);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void EncodeRaw_Matches_HuggingFace_Oracle()
    {
        GptOssTokenizer? tokenizer = TryCreate();
        string? oraclePath = Environment.GetEnvironmentVariable("GPT_OSS_ORACLE_JSON");
        if (tokenizer is null || string.IsNullOrEmpty(oraclePath) || !File.Exists(oraclePath)) return;

        Dictionary<string, int[]> oracle =
            JsonSerializer.Deserialize<Dictionary<string, int[]>>(File.ReadAllText(oraclePath))!;
        foreach ((string text, int[] expected) in oracle)
        {
            IReadOnlyList<int> actual = text switch
            {
                "__template__" => tokenizer.EncodeRaw(
                    GptOssTokenizer.RenderChatTemplate("a photo of a corgi wearing a top hat")),
                "__template_empty__" => tokenizer.EncodeRaw(GptOssTokenizer.RenderChatTemplate("")),
                _ => tokenizer.EncodeRaw(text),
            };
            Assert.Equal(expected, actual.ToArray());
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void BuildChatInputs_Wrapper_Is_Exactly_97_Tokens_And_Unpadded()
    {
        GptOssTokenizer? tokenizer = TryCreate();
        if (tokenizer is null) return;

        (int[] ids, int[] mask) = tokenizer.BuildChatInputs("a photo of a corgi wearing a top hat");
        Assert.Equal(ids.Length, mask.Length);
        Assert.All(mask, m => Assert.Equal(1, m));
        Assert.True(ids.Length < 512, "true-length ids expected, not padded to max length");
        Assert.Equal(200006, ids[0]);
        Assert.Equal(200008, ids[^1]);

        // Position DefaultTxtOffset must be the first user-prompt token ("a" = 64 after "<|message|>").
        (int[] empty, _) = tokenizer.BuildChatInputs("");
        Assert.Equal(119, empty.Length);
    }
}
