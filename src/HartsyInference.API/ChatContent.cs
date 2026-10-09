using System.Text.Json;
using System.Text.Json.Serialization;

namespace HartsyInference.API;

/// <summary>A chat message's <c>content</c>: a plain string, an array of content parts (<c>text</c> and <c>image_url</c>), or null on a tool-call turn. A string converts
/// implicitly, so code that sets text needs no change.</summary>
[JsonConverter(typeof(ChatMessageContentConverter))]
public sealed class ChatMessageContent
{
    /// <summary>The text of a plain-string content. Null when the content is an array of parts.</summary>
    public string? Text { get; init; }

    /// <summary>The parts of an array-form content. Null when the content is a plain string.</summary>
    public IReadOnlyList<ChatContentPart>? Parts { get; init; }

    public static implicit operator ChatMessageContent?(string? text) => text is null ? null : new ChatMessageContent { Text = text };
}

/// <summary>One content part of an array-form message content: <c>{"type":"text","text":...}</c> or <c>{"type":"image_url","image_url":{"url":...}}</c>.</summary>
public sealed class ChatContentPart
{
    public required string Type { get; init; }
    public string? Text { get; init; }

    /// <summary>The image's URL. Only <c>data:</c> URIs are decoded; a remote URL is refused rather than fetched.</summary>
    public string? ImageUrl { get; init; }
}

/// <summary>Reads and writes <see cref="ChatMessageContent"/> in OpenAI's wire shape: a string, or an array of parts that round-trips unchanged.</summary>
internal sealed class ChatMessageContentConverter : JsonConverter<ChatMessageContent>
{
    public override ChatMessageContent? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.String)
            return new ChatMessageContent { Text = reader.GetString() };
        if (reader.TokenType != JsonTokenType.StartArray)
            throw new JsonException("Message content must be a string or an array of content parts.");

        List<ChatContentPart> parts = [];
        while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
        {
            using JsonDocument part = JsonDocument.ParseValue(ref reader);
            parts.Add(ReadPart(part.RootElement));
        }
        return new ChatMessageContent { Parts = parts };
    }

    private static ChatContentPart ReadPart(JsonElement part)
    {
        if (part.ValueKind != JsonValueKind.Object || !part.TryGetProperty("type", out JsonElement type) || type.ValueKind != JsonValueKind.String)
            throw new JsonException("Each content part must be an object with a string 'type'.");
        string? text = part.TryGetProperty("text", out JsonElement t) && t.ValueKind == JsonValueKind.String ? t.GetString() : null;
        string? url = null;
        if (part.TryGetProperty("image_url", out JsonElement image))
        {
            if (image.ValueKind == JsonValueKind.Object && image.TryGetProperty("url", out JsonElement u) && u.ValueKind == JsonValueKind.String)
                url = u.GetString();
            else if (image.ValueKind == JsonValueKind.String)
                url = image.GetString();
        }
        return new ChatContentPart { Type = type.GetString()!, Text = text, ImageUrl = url };
    }

    public override void Write(Utf8JsonWriter writer, ChatMessageContent value, JsonSerializerOptions options)
    {
        if (value.Parts is null)
        {
            writer.WriteStringValue(value.Text);
            return;
        }
        writer.WriteStartArray();
        foreach (ChatContentPart part in value.Parts)
        {
            writer.WriteStartObject();
            writer.WriteString("type", part.Type);
            if (part.Text is not null)
                writer.WriteString("text", part.Text);
            if (part.ImageUrl is not null)
            {
                writer.WriteStartObject("image_url");
                writer.WriteString("url", part.ImageUrl);
                writer.WriteEndObject();
            }
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
    }
}

/// <summary>OpenAI <c>stream_options</c>. <c>include_usage</c> asks a streamed reply to end with a usage frame before <c>[DONE]</c>.</summary>
public sealed class ChatStreamOptionsDto
{
    [JsonPropertyName("include_usage")] public bool IncludeUsage { get; set; }
}

/// <summary>One streamed fragment of a tool call (OpenAI's <c>delta.tool_calls</c> entry). The first fragment of a call carries its <c>id</c>, <c>type</c> and function
/// name; later fragments carry only argument text, which the client concatenates.</summary>
public sealed class ChatToolCallDeltaDto
{
    [JsonPropertyName("index")] public required int Index { get; init; }
    [JsonPropertyName("id")] [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] public string? Id { get; init; }
    [JsonPropertyName("type")] [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] public string? Type { get; init; }
    [JsonPropertyName("function")] public required ChatToolCallDeltaFunctionDto Function { get; init; }
}

public sealed class ChatToolCallDeltaFunctionDto
{
    [JsonPropertyName("name")] [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] public string? Name { get; init; }
    [JsonPropertyName("arguments")] [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] public string? Arguments { get; init; }
}
