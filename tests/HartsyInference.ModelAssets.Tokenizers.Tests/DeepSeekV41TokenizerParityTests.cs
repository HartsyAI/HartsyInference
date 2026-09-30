using System.Text.Json;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ModelAssets.Tokenizers.Tests;

/// <summary>Token-id parity with HuggingFace <c>tokenizers</c> 0.23.2 on the real DeepSeek-V4.1 tokenizer.json (6.3 MB, not committed). Skips unless the file is reachable; set DSV41_TOKENIZER_JSON to point at it.</summary>
[Trait("Category", "Integration")]
public sealed class DeepSeekV41TokenizerParityTests
{
    public const string BosLiteral = "<｜begin▁of▁sentence｜>";
    public const string EosLiteral = "<｜end▁of▁sentence｜>";

    private readonly ITestOutputHelper _output;

    public DeepSeekV41TokenizerParityTests(ITestOutputHelper output) => _output = output;

    private GgufTokenizer? Load()
    {
        string? path = DeepSeekV41ReferenceFiles.FindTokenizerJson();
        if (path is null)
        {
            _output.WriteLine("SKIPPED: DeepSeek-V4.1 tokenizer.json not found (set DSV41_TOKENIZER_JSON).");
            return null;
        }
        using FileStream stream = File.OpenRead(path);
        return HfTokenizerJson.LoadByteLevelBpe(stream, bosToken: BosLiteral, eosToken: EosLiteral);
    }

    private static int[] Ids(JsonElement array) => array.EnumerateArray().Select(e => e.GetInt32()).ToArray();

    [Fact]
    public void StressCorpus_IdsMatchHuggingFace()
    {
        GgufTokenizer? tokenizer = Load();
        if (tokenizer is null) return;
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(DeepSeekV41ReferenceFiles.ReferenceJson));
        int index = 0;
        foreach (JsonElement entry in doc.RootElement.GetProperty("stress").EnumerateArray())
        {
            string text = entry.GetProperty("text").GetString()!;
            Assert.True(Ids(entry.GetProperty("ids")).SequenceEqual(tokenizer.Encode(text, addSpecial: true)),
                $"id mismatch for stress text #{index}: {text}");
            index++;
        }
        Assert.True(index > 50);
    }

    [Fact]
    public void SpecialIds_AreResolved()
    {
        GgufTokenizer? tokenizer = Load();
        if (tokenizer is null) return;
        Assert.Equal(0, tokenizer.BosId);
        Assert.Equal(1, tokenizer.EosId);
        Assert.Equal(129264, tokenizer.SpecialId("<｜deepseek_image｜>"));
    }
}
