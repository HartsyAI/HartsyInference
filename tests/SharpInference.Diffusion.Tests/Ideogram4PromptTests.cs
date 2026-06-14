using Xunit;
using SharpInference.Core.Exceptions;
using SharpInference.Diffusion.Prompting;
using SharpInference.Diffusion.Prompting.Dialects;

namespace SharpInference.Diffusion.Tests;

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
        Assert.Throws<SharpInferenceException>(() => _dialect.Serialize(prompt));
    }

    [Fact]
    public void Validate_RejectsMissingBackground()
    {
        StructuredPrompt prompt = new() { Summary = "x" };
        Assert.Throws<SharpInferenceException>(() => _dialect.Serialize(prompt));
    }

    [Theory]
    [InlineData("#1b1b2")]   // too short
    [InlineData("#GGGGGG")]  // not hex
    [InlineData("1B1B2F")]   // missing '#'
    public void Validate_RejectsBadHex(string color)
    {
        StructuredPrompt prompt = new() { Background = "bg", ColorPalette = [color] };
        Assert.Throws<SharpInferenceException>(() => _dialect.Serialize(prompt));
    }

    [Fact]
    public void Validate_RejectsTooManyElementColors()
    {
        StructuredPrompt prompt = new()
        {
            Background = "bg",
            Elements = [new ObjectElement { Description = "x", ColorPalette = ["#000000", "#111111", "#222222", "#333333", "#444444", "#555555"] }],
        };
        Assert.Throws<SharpInferenceException>(() => _dialect.Serialize(prompt));
    }

    [Fact]
    public void BoundingBox_Validate_RejectsOutOfRangeAndInverted()
    {
        Assert.Throws<SharpInferenceException>(() => new BoundingBox(0, 0, 1001, 500).Validate());
        Assert.Throws<SharpInferenceException>(() => new BoundingBox(500, 0, 400, 500).Validate());
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
