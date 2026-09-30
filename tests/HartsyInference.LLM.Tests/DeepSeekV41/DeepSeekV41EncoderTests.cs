using System.Text.Json;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>Render-text parity of <see cref="DeepSeekV41Encoder"/> against the upstream golden fixtures (test_output_N.txt, byte for byte) and against the upstream encoding.py output for the extra cases in encoder_reference.json.</summary>
public sealed class DeepSeekV41EncoderTests
{
    private static JsonDocument LoadReference() => JsonDocument.Parse(File.ReadAllText(DeepSeekV41ReferenceFiles.ReferenceJson));

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void GoldenFixture_RendersByteEqual(int caseId)
    {
        using JsonDocument input = JsonDocument.Parse(
            File.ReadAllText(Path.Combine(DeepSeekV41ReferenceFiles.EncodingDir, $"test_input_{caseId}.json")));
        string expected = File.ReadAllText(Path.Combine(DeepSeekV41ReferenceFiles.EncodingDir, $"test_output_{caseId}.txt"));
        using JsonDocument reference = LoadReference();
        JsonElement golden = reference.RootElement.GetProperty("golden").EnumerateArray().First(g => g.GetProperty("id").GetInt32() == caseId);
        List<ImageGrid> grids = DeepSeekV41CaseLoader.ReadGrids(golden.GetProperty("grids"));

        JsonElement root = input.RootElement;
        (List<ChatMessage> messages, EncodeOptions options) = DeepSeekV41CaseLoader.Load(root, grids);
        Assert.Equal(expected, DeepSeekV41Encoder.RenderText(messages, options));
    }

    [Fact]
    public void OwnCases_RenderLikeUpstreamEncodingPy()
    {
        using JsonDocument reference = LoadReference();
        int count = 0;
        foreach (JsonElement testCase in reference.RootElement.GetProperty("own").EnumerateArray())
        {
            List<ImageGrid> grids = DeepSeekV41CaseLoader.ReadGrids(testCase.GetProperty("grids"));
            (List<ChatMessage> messages, EncodeOptions options) = DeepSeekV41CaseLoader.Load(testCase, grids);
            string name = testCase.GetProperty("name").GetString()!;
            Assert.True(testCase.GetProperty("prompt").GetString() == DeepSeekV41Encoder.RenderText(messages, options),
                $"rendered text differs for case '{name}'");
            count++;
        }
        Assert.True(count >= 30);
    }

    [Fact]
    public void Encode_ExpandsImagePlaceholdersIntoSpans()
    {
        StubTokenizer tokenizer = new();
        List<ChatMessage> messages =
        [
            ChatMessage.User("before") with { Blocks = [new TextBlock("a"), new ImageBlock(1), new TextBlock("b"), new ImageBlock(0)] },
        ];
        EncodeOptions options = new() { Images = [new ImageGrid(2, 1), new ImageGrid(3, 2)] };

        EncodedConversation encoded = new DeepSeekV41Encoder().Encode(tokenizer, messages, options);

        Assert.Equal(2, encoded.ImageSpans.Count);
        Assert.Equal(1, encoded.ImageSpans[0].ImageIndex);
        Assert.Equal(new ImageGrid(3, 2), encoded.ImageSpans[0].Grid);
        Assert.Equal(2 * (3 + 1) + 2, encoded.ImageSpans[0].Length);
        Assert.Equal(0, encoded.ImageSpans[1].ImageIndex);
        Assert.Equal(new ImageGrid(2, 1), encoded.ImageSpans[1].Grid);
        Assert.Equal(1 * (2 + 1) + 2, encoded.ImageSpans[1].Length);
        Assert.Equal(encoded.Ids.Length, encoded.PromptLength);
        Assert.Equal(encoded.Ids.Length, encoded.DeadMask.Length);
        Assert.Equal(encoded.DeadMask, encoded.VisionRouteMask);
        foreach (ImageSpan span in encoded.ImageSpans)
        {
            for (int i = 0; i < span.Length; i++)
            {
                Assert.Equal(StubTokenizer.PlaceholderId, encoded.Ids[span.Start + i]);
                Assert.True(encoded.DeadMask[span.Start + i]);
            }
        }
        int masked = encoded.DeadMask.Count(m => m);
        Assert.Equal(encoded.ImageSpans.Sum(s => s.Length), masked);
        Assert.DoesNotContain(true, encoded.DeadMask.Take(encoded.ImageSpans[0].Start));
    }

    [Fact]
    public void Encode_ReportsOpenReasoningOnlyWhenPromptEndsInThink()
    {
        StubTokenizer tokenizer = new();
        DeepSeekV41Encoder encoder = new();
        List<ChatMessage> messages = [ChatMessage.User("hi")];
        Assert.True(encoder.Encode(tokenizer, messages, new EncodeOptions { Thinking = true }).InitialParserState.ReasoningOpen);
        Assert.False(encoder.Encode(tokenizer, messages, new EncodeOptions { Thinking = false }).InitialParserState.ReasoningOpen);
        Assert.False(encoder.Encode(tokenizer, messages, new EncodeOptions { AddGenerationPrompt = false, Thinking = true })
            .InitialParserState.ReasoningOpen);
    }

    [Fact]
    public void AddGenerationPromptFalse_OmitsTheAssistantHeader()
    {
        List<ChatMessage> messages = [ChatMessage.User("hi")];
        string with = DeepSeekV41Encoder.RenderText(messages, new EncodeOptions());
        string without = DeepSeekV41Encoder.RenderText(messages, new EncodeOptions { AddGenerationPrompt = false });
        Assert.Equal("<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>", with);
        Assert.Equal("<｜begin▁of▁sentence｜><｜User｜>hi", without);
    }

    [Fact]
    public void Adapter_ReturnsIdsAndMapsThinkingFlag()
    {
        StubTokenizer tokenizer = new();
        ChatTemplateEncoderAdapter adapter = new(new DeepSeekV41Encoder());
        int[] ids = adapter.Encode(tokenizer, [ChatMessage.User("hi")], addGenerationPrompt: true, enableThinking: true);
        Assert.Equal("deepseek_v41", adapter.Name);
        Assert.EndsWith("<｜Assistant｜><think>", tokenizer.LastText);
        Assert.Equal(tokenizer.Encode(tokenizer.LastText!, true), ids);
        adapter.Encode(tokenizer, [ChatMessage.User("hi")], addGenerationPrompt: true);
        Assert.EndsWith("<｜Assistant｜></think>", tokenizer.LastText);
    }

    [Fact]
    public void EffortNames_MapToNumericBudgets()
    {
        Assert.Equal(50, EncodeOptions.ParseReasoningEffort("low"));
        Assert.Equal(75, EncodeOptions.ParseReasoningEffort("high"));
        Assert.Equal(100, EncodeOptions.ParseReasoningEffort("max"));
        Assert.Throws<ArgumentException>(() => EncodeOptions.ParseReasoningEffort("extreme"));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(101)]
    [InlineData(-5)]
    public void OutOfRangeEffort_Throws(int effort)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            DeepSeekV41Encoder.RenderText([ChatMessage.User("x")], new EncodeOptions { Thinking = true, ReasoningEffort = effort }));
    }

    [Fact]
    public void ImagePlaceholderInText_Throws()
    {
        string literal = StubTokenizer.Placeholder;
        Assert.Throws<ArgumentException>(() => DeepSeekV41Encoder.RenderText([ChatMessage.User("a" + literal)], new EncodeOptions()));
        Assert.Throws<ArgumentException>(() => DeepSeekV41Encoder.RenderText(
            [ChatMessage.User("x") with { Blocks = [new TextBlock(literal)] }], new EncodeOptions()));
        Assert.Throws<ArgumentException>(() => DeepSeekV41Encoder.RenderText(
            [ChatMessage.Assistant("x") with { ReasoningContent = literal }], new EncodeOptions()));
    }

    [Fact]
    public void ImagePlaceholderInToolCallArguments_Throws()
    {
        ChatMessage call = ChatMessage.Assistant("x") with
        {
            ToolCalls = [new ChatToolCall("c1", "lookup", "{\"q\":\"" + StubTokenizer.Placeholder + "\"}")],
        };
        Assert.Throws<ArgumentException>(() => DeepSeekV41Encoder.RenderText([ChatMessage.User("q"), call], new EncodeOptions()));
    }

    [Fact]
    public void EmptyOptionTools_KeepFirstMessageTools()
    {
        ToolSpec tool = ToolSpec.FromJson("""{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}""");
        List<ChatMessage> messages = [ChatMessage.System("s") with { Tools = [tool] }, ChatMessage.User("q")];
        string expected = DeepSeekV41Encoder.RenderText(messages, new EncodeOptions());
        Assert.Contains("\"name\": \"lookup\"", expected);
        Assert.Equal(expected, DeepSeekV41Encoder.RenderText(messages, new EncodeOptions { Tools = [] }));
    }

    private const string ConflictingNamespaceTool =
        """{"type":"function","function":{"name":"search::a"},"namespace":"files"}""";

    [Fact]
    public void InvalidInputs_Throw()
    {
        Assert.Throws<ArgumentException>(() => DeepSeekV41Encoder.RenderText(
            [ChatMessage.User("x") with { Task = "nope" }], new EncodeOptions()));
        Assert.Throws<NotSupportedException>(() => DeepSeekV41Encoder.RenderText([new ChatMessage("narrator", "x")], new EncodeOptions()));
        Assert.Throws<ArgumentException>(() => DeepSeekV41Encoder.RenderText(
            [ChatMessage.User("x") with { Blocks = [new ImageBlock(0)] }], new EncodeOptions()));
        Assert.Throws<ArgumentException>(() => DeepSeekV41Encoder.RenderText(
            [ChatMessage.System("s") with { Tools = [ToolSpec.FromJson(ConflictingNamespaceTool)] }],
            new EncodeOptions()));
    }
}
