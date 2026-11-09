using System.Text.Json;
using HartsyInference.API.Endpoints;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Tests;

/// <summary>The OpenAI-shaped <c>video_url</c> content part: parsed, round-tripped and mapped onto the engine's video input.</summary>
public sealed class VideoUrlTests
{
    private static readonly byte[] Clip = [0, 0, 0, 24, (byte)'f', (byte)'t', (byte)'y', (byte)'p'];

    [Fact]
    public void AVideoPartBecomesTheMessagesClip()
    {
        string url = "data:video/mp4;base64," + Convert.ToBase64String(Clip);
        string json = """{"role":"user","content":[{"type":"text","text":"What happens?"},{"type":"video_url","video_url":{"url":"URL"}}]}"""
            .Replace("URL", url, StringComparison.Ordinal);
        ChatMessageDto message = JsonSerializer.Deserialize<ChatMessageDto>(json, new JsonSerializerOptions(JsonSerializerDefaults.Web))!;
        TextMessage native = CompatEndpoints.ToTextMessage(message);
        Assert.Equal("What happens?", native.Content);
        Assert.NotNull(native.Video);
        Assert.Equal(Clip, native.Video!.Data);
        Assert.Equal("mp4", native.Video.Format);

        // The part survives a round trip in OpenAI's shape.
        string back = JsonSerializer.Serialize(message, new JsonSerializerOptions(JsonSerializerDefaults.Web));
        Assert.Contains("\"video_url\":{\"url\":\"data:video/mp4;base64,", back);
    }

    [Theory]
    [InlineData("https://example.com/clip.mp4", "data: URIs")]
    [InlineData("data:video/mp4,AAAA", "base64")]
    [InlineData("data:video/mp4;base64,@@@", "not valid base64")]
    public void ARemoteOrMalformedVideoIsRefused(string url, string expected)
    {
        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => CompatEndpoints.DecodeVideoUrl(url));
        Assert.Contains(expected, error.Message);
    }

    [Fact]
    public void TwoVideosInOneMessageAreRefused()
    {
        string data = "data:video/mp4;base64," + Convert.ToBase64String(Clip);
        ChatMessageDto message = new()
        {
            Role = "user",
            Content = new ChatMessageContent
            {
                Parts = [new ChatContentPart { Type = "video_url", VideoUrl = data }, new ChatContentPart { Type = "video_url", VideoUrl = data }],
            },
        };
        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => CompatEndpoints.ToTextMessage(message));
        Assert.Contains("at most one", error.Message);
    }
}
