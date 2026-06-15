using Xunit;
using HartsyInference.Core.Exceptions;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Diffusion.Prompting.Dialects;
using HartsyInference.Diffusion.Prompting.MagicPrompt;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pure-logic tests for the structured prompt builder + Ideogram 4 JSON dialect (no GPU / checkpoint needed). Verifies the exact serialization Ideogram 4 was trained on: key order, compact separators, literal Unicode, uppercase hex, <c>[y_min,x_min,y_max,x_max]</c> bbox order, the <c>obj</c>/<c>text</c> discriminator, and the validator's mistake-rejection.</summary>
public class Ideogram4PromptTests
{
    private static readonly Ideogram4Dialect _dialect = new();

    [Fact]
    public void Serialize_FullPrompt_ExactJson()
    {
        StructuredPrompt prompt = new StructuredPromptBuilder()
            .Summary("A neon ramen shop at night")
            .Style(s => s.ArtStyle("anime").Lighting("moody neon").Medium("digital illustration"))
            .Background("wet city street")
            .AddObject("a steaming bowl of ramen", new BoundingBox(600, 350, 950, 700), ["#FF6B35", "#f7c59f"])
            .AddText("らーめん", new BoundingBox(50, 100, 200, 900), "glowing red sign")
            .Palette("#FF6B35", "#004E89")
            .Build();

        string json = _dialect.Serialize(prompt);

        // Top-level key order, compact separators, literal unicode, uppercased hex, [y,x,y,x] bbox, obj/text discriminator.
        const string expected =
            "{\"high_level_description\":\"A neon ramen shop at night\"," +
            "\"style_description\":{\"lighting\":\"moody neon\",\"medium\":\"digital illustration\",\"art_style\":\"anime\"}," +
            "\"compositional_deconstruction\":{\"background\":\"wet city street\",\"elements\":[" +
            "{\"type\":\"obj\",\"bbox\":[600,350,950,700],\"desc\":\"a steaming bowl of ramen\",\"color_palette\":[\"#FF6B35\",\"#F7C59F\"]}," +
            "{\"type\":\"text\",\"bbox\":[50,100,200,900],\"text\":\"らーめん\",\"desc\":\"glowing red sign\"}]}}";
        Assert.Equal(expected, json);
    }

    [Fact]
    public void Serialize_MinimalPrompt_OmitsOptionalKeys()
    {
        StructuredPrompt prompt = new() { Background = "a plain studio backdrop" };
        string json = _dialect.Serialize(prompt);
        Assert.Equal(
            "{\"compositional_deconstruction\":{\"background\":\"a plain studio backdrop\",\"elements\":[]}}",
            json);
    }

    [Fact]
    public void Validate_RejectsBothPhotoAndArtStyle()
    {
        StructuredPrompt prompt = new()
        {
            Background = "bg",
            Style = new StyleBlock { Photo = "dslr", ArtStyle = "oil painting" },
        };
        Assert.Throws<HartsyInferenceException>(() => _dialect.Serialize(prompt));
    }

    [Fact]
    public void Validate_RejectsMissingBackground()
    {
        StructuredPrompt prompt = new() { Summary = "x" };
        Assert.Throws<HartsyInferenceException>(() => _dialect.Serialize(prompt));
    }

    [Theory]
    [InlineData("#1b1b2")]   // too short
    [InlineData("#GGGGGG")]  // not hex
    [InlineData("1B1B2F")]   // missing '#'
    public void Validate_RejectsBadHex(string color)
    {
        StructuredPrompt prompt = new() { Background = "bg", ColorPalette = [color] };
        Assert.Throws<HartsyInferenceException>(() => _dialect.Serialize(prompt));
    }

    [Fact]
    public void Validate_RejectsTooManyElementColors()
    {
        StructuredPrompt prompt = new()
        {
            Background = "bg",
            Elements = [new ObjectElement { Description = "x", ColorPalette = ["#000000", "#111111", "#222222", "#333333", "#444444", "#555555"] }],
        };
        Assert.Throws<HartsyInferenceException>(() => _dialect.Serialize(prompt));
    }

    [Fact]
    public void BoundingBox_Validate_RejectsOutOfRangeAndInverted()
    {
        Assert.Throws<HartsyInferenceException>(() => new BoundingBox(0, 0, 1001, 500).Validate());
        Assert.Throws<HartsyInferenceException>(() => new BoundingBox(500, 0, 400, 500).Validate());
        new BoundingBox(0, 0, 1000, 1000).Validate(); // valid — no throw
    }

    [Fact]
    public void BoundingBox_ToPixels_MapsNormalizedToResolution()
    {
        (int x, int y, int w, int h) = new BoundingBox(0, 500, 1000, 1000).ToPixels(1024, 768);
        Assert.Equal(512, x);
        Assert.Equal(0, y);
        Assert.Equal(512, w);
        Assert.Equal(768, h);
    }

    [Fact]
    public void Parse_RoundTrips_FullPrompt()
    {
        // Serialize a rich prompt, parse it back, re-serialize — the JSON must be byte-identical.
        // (Top-level ColorPalette is validated-but-not-serialized, so it legitimately doesn't survive.)
        StructuredPrompt original = new StructuredPromptBuilder()
            .Summary("A neon ramen shop at night")
            .Style(s => s.Aesthetics("cyberpunk").ArtStyle("anime").Lighting("moody neon").Medium("digital illustration"))
            .Background("wet city street")
            .AddObject("a steaming bowl of ramen", new BoundingBox(600, 350, 950, 700), ["#FF6B35", "#F7C59F"])
            .AddText("らーめん", new BoundingBox(50, 100, 200, 900), "glowing red sign")
            .Build();

        string json1 = _dialect.Serialize(original);
        StructuredPrompt parsed = _dialect.Parse(json1);
        string json2 = _dialect.Serialize(parsed);

        Assert.Equal(json1, json2);
    }

    [Fact]
    public void Parse_ReconstructsFields()
    {
        const string json =
            "{\"high_level_description\":\"a cat\"," +
            "\"style_description\":{\"lighting\":\"soft\",\"photo\":\"dslr\",\"color_palette\":[\"#FF0000\"]}," +
            "\"compositional_deconstruction\":{\"background\":\"a room\",\"elements\":[" +
            "{\"type\":\"obj\",\"bbox\":[10,20,30,40],\"desc\":\"a ginger cat\",\"color_palette\":[\"#ABCDEF\"]}," +
            "{\"type\":\"text\",\"text\":\"HELLO\",\"desc\":\"poster\"}]}}";

        StructuredPrompt p = _dialect.Parse(json);

        Assert.Equal("a cat", p.Summary);
        Assert.Equal("soft", p.Style!.Lighting);
        Assert.Equal("dslr", p.Style.Photo);
        Assert.Equal(new[] { "#FF0000" }, p.Style.ColorPalette);
        Assert.Equal("a room", p.Background);
        Assert.Equal(2, p.Elements.Count);
        ObjectElement obj = Assert.IsType<ObjectElement>(p.Elements[0]);
        Assert.Equal("a ginger cat", obj.Description);
        Assert.Equal(new BoundingBox(10, 20, 30, 40), obj.Bbox);
        Assert.Equal(new[] { "#ABCDEF" }, obj.ColorPalette);
        TextElement txt = Assert.IsType<TextElement>(p.Elements[1]);
        Assert.Equal("HELLO", txt.Text);
        Assert.Null(txt.Bbox);
    }

    [Fact]
    public void Parse_RejectsMalformedJsonAndUnknownType()
    {
        Assert.Throws<HartsyInferenceException>(() => _dialect.Parse("{not json"));
        Assert.Throws<HartsyInferenceException>(() => _dialect.Parse("[1,2,3]"));
        Assert.Throws<HartsyInferenceException>(() =>
            _dialect.Parse("{\"compositional_deconstruction\":{\"background\":\"x\",\"elements\":[{\"type\":\"widget\"}]}}"));
    }

    [Fact]
    public async Task DelegateMagicPromptExpander_ParsesLlmJson_AndValidates()
    {
        // Fake LLM that wraps the JSON in a ```json fence + prose (the expander must tolerate it).
        const string llmReply =
            "Sure! Here is the structured prompt:\n```json\n" +
            "{\"high_level_description\":\"a cat on a mat\"," +
            "\"compositional_deconstruction\":{\"background\":\"a sunny room\",\"elements\":[" +
            "{\"type\":\"obj\",\"desc\":\"a ginger cat\"}]}}\n```\nHope that helps!";

        DelegateMagicPromptExpander expander = new(
            systemPrompt: "emit ideogram json",
            complete: (_, _, _) => Task.FromResult(llmReply));

        StructuredPrompt p = await expander.ExpandAsync("a cat");

        Assert.Equal("a cat on a mat", p.Summary);
        Assert.Equal("a sunny room", p.Background);
        Assert.Equal("a ginger cat", Assert.IsType<ObjectElement>(Assert.Single(p.Elements)).Description);
        // Re-serializing the expander's output must produce valid Ideogram JSON.
        Assert.Contains("a cat on a mat", _dialect.Serialize(p));
    }

    [Fact]
    public async Task DelegateMagicPromptExpander_ThrowsOnEmptyOrNonJsonResponse()
    {
        DelegateMagicPromptExpander empty = new("sys", (_, _, _) => Task.FromResult("   "));
        await Assert.ThrowsAsync<HartsyInferenceException>(() => empty.ExpandAsync("x"));

        DelegateMagicPromptExpander prose = new("sys", (_, _, _) => Task.FromResult("no json here"));
        await Assert.ThrowsAsync<HartsyInferenceException>(() => prose.ExpandAsync("x"));
    }

    [Fact]
    public void NaturalLanguageDialect_FlattensToProse()
    {
        StructuredPrompt prompt = new StructuredPromptBuilder()
            .Summary("A cat on a mat")
            .Style(s => s.ArtStyle("watercolor").Lighting("soft"))
            .Background("a sunny room")
            .AddObject("a ginger cat curled up")
            .AddText("HELLO", desc: "wall poster")
            .Build();

        string prose = new NaturalLanguageDialect().Serialize(prompt);
        Assert.Contains("A cat on a mat", prose);
        Assert.Contains("watercolor", prose);
        Assert.Contains("Background: a sunny room", prose);
        Assert.Contains("ginger cat", prose);
        Assert.Contains("\"HELLO\"", prose);
    }
}
