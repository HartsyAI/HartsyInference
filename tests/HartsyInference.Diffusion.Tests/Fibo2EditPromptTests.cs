using System.Text.Json.Nodes;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Fibo 2.0 edit prompt: ComfyUI's <c>tokenize_with_weights</c> with images (the JSON wrapper and its marker per image) and
/// <c>parse_prompt</c> (caption cleaning), checked on the strings those functions produce.</summary>
public sealed class Fibo2EditPromptTests
{
    private const string Marker = "<|vision_start|><|image_pad|><|vision_end|>";

    [Fact]
    public void Build_PlainInstruction_WrapsOneImageAsImage()
    {
        string json = Fibo2EditPrompt.Build("replace the jacket with a red leather one", 1);

        Assert.Equal($"{{\"image\":\"{Marker}\",\"edit_instruction\":\"replace the jacket with a red leather one\"}}", json);
    }

    [Fact]
    public void Build_SeveralImages_NumbersTheMarkers()
    {
        string json = Fibo2EditPrompt.Build("dress <image_1> in the dress from <image_2>", 2);

        Assert.Equal($"{{\"image_1\":\"{Marker}\",\"image_2\":\"{Marker}\","
            + "\"edit_instruction\":\"dress <image_1> in the dress from <image_2>\"}", json);
    }

    [Fact]
    public void Build_CaptionWithInstruction_KeepsTheCleanedCaptionAsStructuredCaption()
    {
        string prompt = "{\"short_description\":\"a cat\",\"objects\":[],\"lighting\":{\"aesthetic_score\":3,\"direction\":\"\"},"
            + "\"aesthetic_score\":9,\"edit_instruction\":\"make it a dog\"}";

        string json = Fibo2EditPrompt.Build(prompt, 1);

        Assert.Equal($"{{\"image\":\"{Marker}\",\"structured_caption\":{{\"short_description\":\"a cat\"}},"
            + "\"edit_instruction\":\"make it a dog\"}", json);
    }

    [Fact]
    public void Build_JsonWithoutInstruction_IsTheInstructionVerbatim()
    {
        // ComfyUI: with images and no edit_instruction key, the whole prompt text becomes the instruction.
        string json = Fibo2EditPrompt.Build("{\"short_description\":\"a cat\"}", 1);

        JsonNode edit = JsonNode.Parse(json)!;
        Assert.Equal("{\"short_description\":\"a cat\"}", (string?)edit["edit_instruction"]);
        Assert.Null(edit["structured_caption"]);
    }

    [Fact]
    public void Build_KeepsNonAsciiAndAngleBracketsUnescaped()
    {
        string json = Fibo2EditPrompt.Build("ändere <image_1> à « café »", 1);

        Assert.Contains("ändere <image_1> à « café »", json, StringComparison.Ordinal);
    }

    [Fact]
    public void Parse_ShortDescriptionAndTextRender_KeepBothKeysEvenWhenEmpty()
    {
        (JsonNode? caption, JsonNode? instruction) = Fibo2EditPrompt.Parse("{\"short_description\":\"\",\"text_render\":[]}");

        Assert.Null(instruction);
        Assert.Equal("{\"short_description\":\"\",\"text_render\":[]}", caption!.ToJsonString());
    }

    [Fact]
    public void Parse_StructuredCaptionKey_IsUnwrapped()
    {
        (JsonNode? caption, JsonNode? instruction) = Fibo2EditPrompt.Parse(
            "{\"structured_caption\":{\"short_description\":\"x\",\"preference_score\":1},\"edit_instruction\":\"do it\"}");

        Assert.Equal("{\"short_description\":\"x\"}", caption!.ToJsonString());
        Assert.Equal("do it", (string?)instruction);
    }

    [Fact]
    public void Parse_NonObject_ComesBackAsItself()
    {
        (JsonNode? caption, JsonNode? instruction) = Fibo2EditPrompt.Parse("make the sky pink");

        Assert.Equal("make the sky pink", (string?)caption);
        Assert.Null(instruction);
    }

    [Fact]
    public void Build_GlyphTextCaption_IsRefused()
    {
        string prompt = "{\"short_description\":\"a sign\",\"text_render\":[{\"text\":\"OPEN\"}],\"edit_instruction\":\"paint it\"}";

        Assert.Throws<NotSupportedException>(() => Fibo2EditPrompt.Build(prompt, 1));
    }

    [Fact]
    public void Tokenize_ExpandsEachMarkerToItsVisionTokens()
    {
        using Qwen3Tokenizer tokenizer = new();
        string json = Fibo2EditPrompt.Build("make it snow around <image_1> and <image_2>", 2);

        int[] ids = Fibo2EditPrompt.Tokenize(tokenizer, json, [3, 5]);

        Assert.Equal(Qwen3Tokenizer.ImStartId, ids[0]);
        Assert.Equal(3 + 5, ids.Count(id => id == Fibo2EditPrompt.ImagePadId));
        Assert.Equal(2, ids.Count(id => id == Fibo2EditPrompt.VisionStartId));
        Assert.Equal(2, ids.Count(id => id == Fibo2EditPrompt.VisionEndId));
        int first = Array.IndexOf(ids, Fibo2EditPrompt.VisionStartId);
        Assert.Equal(Enumerable.Repeat(Fibo2EditPrompt.ImagePadId, 3), ids.Skip(first + 1).Take(3));
        Assert.Equal(Fibo2EditPrompt.VisionEndId, ids[first + 4]);
        // The template: user turn, assistant opener, no think block.
        int[] tail = [Qwen3Tokenizer.ImEndId, .. tokenizer.EncodeRaw("\n"), Qwen3Tokenizer.ImStartId, .. tokenizer.EncodeRaw("assistant\n")];
        Assert.Equal(tail, ids.Skip(ids.Length - tail.Length));
    }

    [Fact]
    public void Tokenize_MarkerCountMustMatchTheImages()
    {
        using Qwen3Tokenizer tokenizer = new();
        string json = Fibo2EditPrompt.Build("edit", 1);

        Assert.Throws<ArgumentException>(() => Fibo2EditPrompt.Tokenize(tokenizer, json, [2, 2]));
    }
}
