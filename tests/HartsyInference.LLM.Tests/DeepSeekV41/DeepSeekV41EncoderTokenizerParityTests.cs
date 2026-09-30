using System.Text.Json;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>Token ids of <see cref="DeepSeekV41Encoder"/> on the real tokenizer.json (not committed) against HuggingFace <c>tokenizers</c> ids of the upstream-rendered prompt with image placeholders expanded.</summary>
[Trait("Category", "Integration")]
public sealed class DeepSeekV41EncoderTokenizerParityTests
{
    private readonly ITestOutputHelper _output;

    public DeepSeekV41EncoderTokenizerParityTests(ITestOutputHelper output) => _output = output;

    private GgufTokenizer? LoadTokenizer()
    {
        string? path = DeepSeekV41ReferenceFiles.FindTokenizerJson();
        if (path is null)
        {
            _output.WriteLine("SKIPPED: DeepSeek-V4.1 tokenizer.json not found (set DSV41_TOKENIZER_JSON).");
            return null;
        }
        using FileStream stream = File.OpenRead(path);
        return HfTokenizerJson.LoadByteLevelBpe(stream,
            bosToken: "<｜begin▁of▁sentence｜>", eosToken: "<｜end▁of▁sentence｜>");
    }

    private static int[] Ids(JsonElement array) => array.EnumerateArray().Select(e => e.GetInt32()).ToArray();

    private static void AssertSpansConsistent(EncodedConversation encoded, int placeholderId)
    {
        Assert.Equal(encoded.DeadMask, encoded.VisionRouteMask);
        Assert.Equal(encoded.Ids.Length, encoded.PromptLength);
        bool[] expectedMask = new bool[encoded.Ids.Length];
        foreach (ImageSpan span in encoded.ImageSpans)
        {
            Assert.Equal(span.Grid.TokenCount, span.Length);
            for (int i = 0; i < span.Length; i++)
            {
                Assert.Equal(placeholderId, encoded.Ids[span.Start + i]);
                expectedMask[span.Start + i] = true;
            }
        }
        Assert.Equal(expectedMask, encoded.DeadMask);
    }

    [Fact]
    public void GoldenFixtures_IdsMatchHuggingFace()
    {
        GgufTokenizer? tokenizer = LoadTokenizer();
        if (tokenizer is null) return;
        using JsonDocument reference = JsonDocument.Parse(File.ReadAllText(DeepSeekV41ReferenceFiles.ReferenceJson));
        int checkedCases = 0;
        foreach (JsonElement golden in reference.RootElement.GetProperty("golden").EnumerateArray())
        {
            int id = golden.GetProperty("id").GetInt32();
            using JsonDocument input = JsonDocument.Parse(
                File.ReadAllText(Path.Combine(DeepSeekV41ReferenceFiles.EncodingDir, $"test_input_{id}.json")));
            List<ImageGrid> grids = DeepSeekV41CaseLoader.ReadGrids(golden.GetProperty("grids"));
            (List<ChatMessage> messages, EncodeOptions options) = DeepSeekV41CaseLoader.Load(input.RootElement, grids);
            EncodedConversation encoded = new DeepSeekV41Encoder().Encode(tokenizer, messages, options);
            Assert.True(Ids(golden.GetProperty("ids")).SequenceEqual(encoded.Ids), $"ids differ for golden case {id}");
            AssertSpansConsistent(encoded, 129264);
            checkedCases++;
        }
        Assert.Equal(5, checkedCases);
    }

    [Fact]
    public void OwnCases_IdsMatchHuggingFace()
    {
        GgufTokenizer? tokenizer = LoadTokenizer();
        if (tokenizer is null) return;
        using JsonDocument reference = JsonDocument.Parse(File.ReadAllText(DeepSeekV41ReferenceFiles.ReferenceJson));
        int checkedCases = 0;
        foreach (JsonElement testCase in reference.RootElement.GetProperty("own").EnumerateArray())
        {
            List<ImageGrid> grids = DeepSeekV41CaseLoader.ReadGrids(testCase.GetProperty("grids"));
            (List<ChatMessage> messages, EncodeOptions options) = DeepSeekV41CaseLoader.Load(testCase, grids);
            EncodedConversation encoded = new DeepSeekV41Encoder().Encode(tokenizer, messages, options);
            string name = testCase.GetProperty("name").GetString()!;
            Assert.True(Ids(testCase.GetProperty("ids")).SequenceEqual(encoded.Ids), $"ids differ for case '{name}'");
            AssertSpansConsistent(encoded, 129264);
            Assert.Equal(grids.Count > 0, encoded.ImageSpans.Count > 0);
            checkedCases++;
        }
        Assert.True(checkedCases >= 30);
    }
}
