using System.Text.Json;
using System.Text.Json.Serialization;
using HartsyInference.API;
using HartsyInference.API.Endpoints;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>The chat DTOs in OpenAI's wire shape: content as a string, as parts, or null; <c>reasoning_content</c>; <c>stream_options</c>; and the mapping of image
/// parts to native images (only <c>data:</c> URIs are decoded; a remote URL is refused).</summary>
public sealed class ChatDtoRoundTripTests
{
    private static readonly JsonSerializerOptions s_options = new(JsonSerializerDefaults.Web) { Converters = { new JsonStringEnumConverter() } };

    /// <summary>A 1×1 RGBA PNG, base64.</summary>
    private const string OnePixelPng = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

    [Fact]
    public void String_Content_Reads_As_Text()
    {
        ChatMessageDto m = JsonSerializer.Deserialize<ChatMessageDto>("""{"role":"user","content":"hello"}""", s_options)!;

        Assert.Equal("hello", m.Content!.Text);
        Assert.Null(m.Content.Parts);
    }

    [Fact]
    public void Content_Parts_Round_Trip_Unchanged()
    {
        const string content = """[{"type":"text","text":"What is this?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]""";
        ChatMessageDto m = JsonSerializer.Deserialize<ChatMessageDto>($$"""{"role":"user","content":{{content}}}""", s_options)!;

        Assert.Equal(2, m.Content!.Parts!.Count);
        Assert.Equal("data:image/png;base64,AAAA", m.Content.Parts[1].ImageUrl);
        using JsonDocument written = JsonDocument.Parse(JsonSerializer.Serialize(m, s_options));
        JsonElement parts = written.RootElement.GetProperty("content");
        Assert.Equal(JsonValueKind.Array, parts.ValueKind);
        Assert.Equal("What is this?", parts[0].GetProperty("text").GetString());
        Assert.Equal("image_url", parts[1].GetProperty("type").GetString());
        Assert.Equal("data:image/png;base64,AAAA", parts[1].GetProperty("image_url").GetProperty("url").GetString());
    }

    [Fact]
    public void Null_Content_On_A_Tool_Call_Turn_Stays_Null()
    {
        ChatMessageDto m = JsonSerializer.Deserialize<ChatMessageDto>("""{"role":"assistant","content":null}""", s_options)!;

        Assert.Null(m.Content);
        using JsonDocument written = JsonDocument.Parse(JsonSerializer.Serialize(m, s_options));
        Assert.Equal(JsonValueKind.Null, written.RootElement.GetProperty("content").ValueKind);
    }

    [Fact]
    public void Reasoning_Content_And_Stream_Options_Parse()
    {
        const string json = """{"model":"m","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"assistant","content":"x","reasoning_content":"thinking"}]}""";

        ChatCompletionRequest req = JsonSerializer.Deserialize<ChatCompletionRequest>(json, s_options)!;

        Assert.True(req.StreamOptions!.IncludeUsage);
        Assert.Equal("thinking", req.Messages[0].ReasoningContent);
    }

    [Fact]
    public void Text_And_Image_Parts_Map_To_One_Native_Message_With_Its_Image_And_Reasoning()
    {
        ChatMessageDto m = new()
        {
            Role = "user",
            Content = new ChatMessageContent
            {
                Parts =
                [
                    new ChatContentPart { Type = "text", Text = "Describe " },
                    new ChatContentPart { Type = "text", Text = "this." },
                    new ChatContentPart { Type = "image_url", ImageUrl = $"data:image/png;base64,{OnePixelPng}" },
                ],
            },
            ReasoningContent = "carried back",
        };

        TextMessage native = CompatEndpoints.ToTextMessage(m);

        Assert.Equal("Describe this.", native.Content);
        HartsyInference.Engine.Requests.ImageData image = Assert.Single(native.Images!);
        Assert.Equal(1, image.Width);
        Assert.Equal(1, image.Height);
        Assert.Equal("carried back", native.ReasoningContent);
    }

    [Fact]
    public void A_Remote_Image_URL_Is_Refused_Rather_Than_Fetched()
    {
        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => CompatEndpoints.DecodeImageUrl("https://example.com/cat.png"));

        Assert.Contains("data: URIs", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void An_Image_Over_32_MiB_Decoded_Is_Refused_Before_It_Is_Decoded()
    {
        const string prefix = "data:image/png;base64,";
        // One base64 quantum (3 bytes) past the limit.
        int payload = (CompatEndpoints.MaxImageBytes / 3 + 1) * 4;
        string url = string.Create(prefix.Length + payload, prefix, static (span, head) =>
        {
            head.AsSpan().CopyTo(span);
            span[head.Length..].Fill('A');
        });

        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => CompatEndpoints.DecodeImageUrl(url));

        Assert.Equal("Image data decodes to about 33554433 bytes, over the 33554432-byte (32 MiB) limit.", ex.Message);
    }

    [Fact]
    public void An_Unknown_Content_Part_Type_Is_Refused()
    {
        ChatMessageDto m = new()
        {
            Role = "user",
            Content = new ChatMessageContent { Parts = [new ChatContentPart { Type = "input_audio" }] },
        };

        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => CompatEndpoints.ToTextMessage(m));

        Assert.Contains("input_audio", ex.Message, StringComparison.Ordinal);
    }
}
