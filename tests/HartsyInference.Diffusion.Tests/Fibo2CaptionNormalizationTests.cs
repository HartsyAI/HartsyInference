using HartsyInference.Diffusion.Prompting;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Fibo 2.0 caption handling against ComfyUI's <c>parse_prompt</c> and <c>tokenize_with_weights</c>
/// (commit 87ed5a066d9c). The expected strings were produced by running ComfyUI's own functions on the inputs.</summary>
public sealed class Fibo2CaptionNormalizationTests
{
    [Theory]
    [InlineData("""a red bicycle""", """a red bicycle""", """a red bicycle""")]
    [InlineData("""{ "short_description": "a sign", "objects": [], "aesthetic_score": 7.5, "lighting": {"conditions":"soft", "shadows": "", "preference_score": 1}, "context": null }""", """{"short_description":"a sign","lighting":{"conditions":"soft"}}""", """{"short_description":"a sign","lighting":{"conditions":"soft"}}""")]
    [InlineData("""{"short_description":"x","text_render":[]}""", """{"short_description":"x","text_render":[]}""", """{"short_description":"x","text_render":[]}""")]
    [InlineData("""{"short_description":"","text_render":[]}""", """{"short_description":"","text_render":[]}""", """{"short_description":"","text_render":[]}""")]
    [InlineData("""{"short_description":"a sign","text_render":[{"text":"OPEN","color":"red","font":""},{"text":""},{"text":"Café ☃"}]}""", """{"short_description":"a sign","text_render":[{"text":"OPEN","color":"red","image_tokens":""},{"text":"Café ☃","image_tokens":""}]}""", """{"short_description":"a sign","text_render":[{"text":"OPEN","color":"red","image_tokens":"<|vision_start|><|image_pad|><|vision_end|>"},{"text":"Café ☃","image_tokens":"<|vision_start|><|image_pad|><|vision_end|>"}]}""")]
    [InlineData("""{}""", """{}""", """{}""")]
    [InlineData("""{"a":""}""", """{}""", """{}""")]
    [InlineData("""[1,2]""", """[1,2]""", """[1,2]""")]
    [InlineData("""123""", """123""", """123""")]
    [InlineData("""null""", """null""", """null""")]
    [InlineData("""{"structured_caption":{"short_description":"in","objects":[{"description":"cat","x":0.5}]},"edit_instruction":"make it blue"}""", """{"short_description":"in","objects":[{"description":"cat","x":0.5}]}""", """{"short_description":"in","objects":[{"description":"cat","x":0.5}]}""")]
    [InlineData("""{"structured_caption":"hello"}""", """{"structured_caption":"hello"}""", """{"structured_caption":"hello"}""")]
    [InlineData("""{"structured_caption":[1,{"a":""}]}""", """[1,{"a":""}]""", """[1,{"a":""}]""")]
    [InlineData("""{"structured_caption":0}""", """{}""", """{}""")]
    [InlineData("""{"n":[1.0,1e5,1e16,1.5e-7,0.0001,-0.0,12345678901234567890,1E400,-0,100.5e-2,NaN,Infinity,true,false,0]}""", """{"n":[1.0,100000.0,1e+16,1.5e-07,0.0001,-0.0,12345678901234567890,Infinity,0,1.005,Infinity,true,false,0]}""", """{"n":[1.0,100000.0,1e+16,1.5e-07,0.0001,-0.0,12345678901234567890,Infinity,0,1.005,Infinity,true,false,0]}""")]
    [InlineData("""{"short_description":0,"text_render":0}""", """{"short_description":"","text_render":[]}""", """{"short_description":"","text_render":[]}""")]
    [InlineData("""{"k":"a\u0001\n\"q\"\/ \ud83d\ude00 \u00e9"}""", """{"k":"a\u0001\n\"q\"/ 😀 é"}""", """{"k":"a\u0001\n\"q\"/ 😀 é"}""")]
    [InlineData("""{"a":1,"b":2,"a":3}""", """{"a":3,"b":2}""", """{"a":3,"b":2}""")]
    [InlineData("""{bad json""", """{bad json""", """{bad json""")]
    [InlineData("""  {"short_description":"padded"}  """, """{"short_description":"padded"}""", """{"short_description":"padded"}""")]
    [InlineData("""{"text_render":[{"text":"HI"}],"short_description":"s","extra":""}""", """{"text_render":[{"text":"HI","image_tokens":""}],"short_description":"s"}""", """{"text_render":[{"text":"HI","image_tokens":"<|vision_start|><|image_pad|><|vision_end|>"}],"short_description":"s"}""")]
    public void Normalize_MatchesComfyUi(string prompt, string withoutImages, string withImages)
    {
        Assert.Equal(withoutImages, Fibo2PromptTemplate.Normalize(prompt, textImages: false).Text);
        Assert.Equal(withImages, Fibo2PromptTemplate.Normalize(prompt, textImages: true).Text);
    }

    [Theory]
    [InlineData("")]
    [InlineData("  \n")]
    public void Normalize_EmptyPrompt_IsTheNullCaptionUntouched(string prompt)
    {
        Fibo2NormalizedPrompt normalized = Fibo2PromptTemplate.Normalize(prompt, textImages: true);

        Assert.Equal(Fibo2PromptTemplate.NullCaption, normalized.Text);
        Assert.Empty(normalized.GlyphTexts);
    }

    [Fact]
    public void Normalize_NullCaptionAsAPromptCollapsesToEmptyObject()
    {
        Assert.Equal("{}", Fibo2PromptTemplate.Normalize(Fibo2PromptTemplate.NullCaption, textImages: false).Text);
    }

    [Fact]
    public void Normalize_GlyphEntries_ListTheTextOfEachNonEmptyEntry()
    {
        const string prompt = "{\"short_description\":\"a sign\",\"text_render\":[{\"text\":\"OPEN\"},{\"text\":\"\"},{\"text\":\"Café\"}]}";

        Fibo2NormalizedPrompt normalized = Fibo2PromptTemplate.Normalize(prompt, textImages: true);

        Assert.Equal(["OPEN", "Café"], normalized.GlyphTexts);
        Assert.Equal(2, normalized.Text.Split(Fibo2PromptTemplate.VisionMarker).Length - 1);
    }

    [Theory]
    [InlineData(0.0, true)]
    [InlineData(0.9, true)]
    [InlineData(0.9000001, false)]
    [InlineData(1.0, false)]
    public void UsesGlyphImages_GatesAtSigmaPointNine(double sigma, bool expected)
    {
        Assert.Equal(expected, Fibo2PromptTemplate.UsesGlyphImages(sigma));
    }

    /// <summary>ComfyUI's <c>TEXT_IMAGES_START</c> percent maps to sigma 0.9 under the flux shift <c>s = e^1.15</c>.</summary>
    [Fact]
    public void TextImagesStartPercent_MapsToSigmaPointNine()
    {
        double s = Math.Exp(1.15);
        double percent = s / (9 + s);
        double t = 1 - percent;

        Assert.Equal(0.9, s * t / (1 + (s - 1) * t), 12);
    }

    [Fact]
    public void TokenizePrompt_PrettyJsonCaption_HasTheTokensOfItsCompactForm()
    {
        using Qwen3Tokenizer tokenizer = new();
        const string pretty = "{\n  \"short_description\": \"a red bicycle\",\n  \"objects\": [],\n  \"aesthetic_score\": 7.5,\n"
            + "  \"lighting\": {\n    \"conditions\": \"soft\",\n    \"shadows\": \"\"\n  }\n}";
        const string compact = "{\"short_description\":\"a red bicycle\",\"lighting\":{\"conditions\":\"soft\"}}";

        int[] fromPretty = Fibo2PromptTemplate.TokenizePrompt(tokenizer, pretty);

        Assert.Equal(Fibo2PromptTemplate.Tokenize(tokenizer, compact), fromPretty);
        Assert.True(fromPretty.Length < Fibo2PromptTemplate.Tokenize(tokenizer, pretty).Length);
    }

    [Fact]
    public void TokenizePrompt_EmptyPrompt_IsTheNullCaptionTokens()
    {
        using Qwen3Tokenizer tokenizer = new();

        Assert.Equal(Fibo2PromptTemplate.Tokenize(tokenizer, Fibo2PromptTemplate.NullCaption), Fibo2PromptTemplate.TokenizePrompt(tokenizer, " "));
    }

    [Fact]
    public void TokenizePrompt_PlainText_IsTemplatedAsWritten()
    {
        using Qwen3Tokenizer tokenizer = new();

        Assert.Equal(Fibo2PromptTemplate.Tokenize(tokenizer, "  a red bicycle "), Fibo2PromptTemplate.TokenizePrompt(tokenizer, "  a red bicycle "));
    }

    [Fact]
    public void TokenizePrompt_GlyphEntryWithText_IsRefusedNotDropped()
    {
        using Qwen3Tokenizer tokenizer = new();
        const string prompt = "{\"short_description\":\"a sign\",\"text_render\":[{\"text\":\"OPEN\"}]}";

        NotSupportedException error = Assert.Throws<NotSupportedException>(() => Fibo2PromptTemplate.TokenizePrompt(tokenizer, prompt));

        Assert.Contains("glyph text images are not implemented yet for Fibo 2.0", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TokenizePrompt_TextRenderWithoutText_IsAnOrdinaryCaption()
    {
        using Qwen3Tokenizer tokenizer = new();
        const string prompt = "{\"short_description\":\"a sign\",\"text_render\":[{\"text\":\"\"}]}";

        int[] ids = Fibo2PromptTemplate.TokenizePrompt(tokenizer, prompt);

        Assert.Equal(Fibo2PromptTemplate.Tokenize(tokenizer, "{\"short_description\":\"a sign\",\"text_render\":[]}"), ids);
    }

    [Fact]
    public void TokenizeWithImages_SplicesPadsAtTheMarkersAndLeavesTheRestOfTheTemplate()
    {
        using Qwen3Tokenizer tokenizer = new();
        const string prompt = "{\"short_description\":\"s\",\"text_render\":[{\"text\":\"A\"},{\"text\":\"B\"}]}";

        int[] ids = Fibo2PromptTemplate.TokenizeWithImages(tokenizer, prompt, [3, 5]);

        Assert.Equal(Qwen3Tokenizer.ImStartId, ids[0]);
        Assert.Equal(8, ids.Count(id => id == Fibo2PromptTemplate.ImagePadId));
        int first = Array.IndexOf(ids, Fibo2PromptTemplate.VisionStartId);
        Assert.Equal(Fibo2PromptTemplate.ImagePadId, ids[first + 1]);
        Assert.Equal(Fibo2PromptTemplate.VisionEndId, ids[first + 4]);
        int second = Array.IndexOf(ids, Fibo2PromptTemplate.VisionStartId, first + 1);
        Assert.Equal(Fibo2PromptTemplate.VisionEndId, ids[second + 6]);
        Assert.Equal(Qwen3Tokenizer.ImEndId, ids[ids.ToList().LastIndexOf(Qwen3Tokenizer.ImEndId)]);

        List<int> expectedHead = [Qwen3Tokenizer.ImStartId];
        expectedHead.AddRange(tokenizer.EncodeRaw("user\n{\"short_description\":\"s\",\"text_render\":[{\"text\":\"A\",\"image_tokens\":\""));
        Assert.Equal(expectedHead, ids.Take(expectedHead.Count));
    }

    [Fact]
    public void TokenizeWithImages_GrowsByThePadsAndMarkerTokensOverTheWithoutImagesPass()
    {
        using Qwen3Tokenizer tokenizer = new();
        const string prompt = "{\"short_description\":\"s\",\"text_render\":[{\"text\":\"A\"}]}";

        int[] without = Fibo2PromptTemplate.TokenizeWithoutImages(tokenizer, prompt);
        int[] with = Fibo2PromptTemplate.TokenizeWithImages(tokenizer, prompt, [10]);

        Assert.Equal(without.Length + 12, with.Length);
    }

    [Fact]
    public void TokenizeWithImages_PadCountMismatch_Throws()
    {
        using Qwen3Tokenizer tokenizer = new();
        const string prompt = "{\"short_description\":\"s\",\"text_render\":[{\"text\":\"A\"}]}";

        Assert.Throws<ArgumentException>(() => Fibo2PromptTemplate.TokenizeWithImages(tokenizer, prompt, [1, 1]));
    }
}
