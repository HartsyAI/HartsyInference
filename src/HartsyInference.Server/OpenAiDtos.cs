using System.Text.Json.Serialization;

namespace HartsyInference.Server;

/// <summary>OpenAI-compatible request/response DTOs. Field names use snake_case via attributes to match
/// the OpenAI Images / error wire format so the official client SDKs interoperate.</summary>
public sealed class ImageGenerationRequest
{
    /// <summary>Text prompt.</summary>
    [JsonPropertyName("prompt")] public string Prompt { get; set; } = "";

    /// <summary>Model id (must already be loaded via <c>/v1/models/load</c> or <c>/v1/models/pull</c>).</summary>
    [JsonPropertyName("model")] public string? Model { get; set; }

    /// <summary>Negative prompt (HartsyInference extension).</summary>
    [JsonPropertyName("negative_prompt")] public string? NegativePrompt { get; set; }

    /// <summary>Image size "WxH" (e.g. "1024x1024").</summary>
    [JsonPropertyName("size")] public string? Size { get; set; }

    /// <summary>Number of images to generate.</summary>
    [JsonPropertyName("n")] public int N { get; set; } = 1;

    /// <summary>Denoise steps (HartsyInference extension).</summary>
    [JsonPropertyName("steps")] public int Steps { get; set; } = 28;

    /// <summary>Classifier-free guidance scale (HartsyInference extension).</summary>
    [JsonPropertyName("cfg_scale")] public float CfgScale { get; set; } = 7.0f;

    /// <summary>Seed (-1 = random).</summary>
    [JsonPropertyName("seed")] public long Seed { get; set; } = -1;

    /// <summary>CLIP skip (HartsyInference extension).</summary>
    [JsonPropertyName("clip_skip")] public int? ClipSkip { get; set; }

    /// <summary>Response format: "b64_json" (only supported value; "url" is not, since the server is stateless).</summary>
    [JsonPropertyName("response_format")] public string ResponseFormat { get; set; } = "b64_json";
}

/// <summary>One generated image (base64 PNG).</summary>
public sealed class ImageData
{
    [JsonPropertyName("b64_json")] public required string B64Json { get; init; }
}

/// <summary>OpenAI images response envelope.</summary>
public sealed class ImageGenerationResponse
{
    [JsonPropertyName("created")] public required long Created { get; init; }
    [JsonPropertyName("data")] public required IReadOnlyList<ImageData> Data { get; init; }
}

/// <summary>One entry in the models list.</summary>
public sealed class ModelEntry
{
    [JsonPropertyName("id")] public required string Id { get; init; }
    [JsonPropertyName("object")] public string Object => "model";
    [JsonPropertyName("owned_by")] public string OwnedBy => "hartsyinference";
}

/// <summary>OpenAI models list envelope.</summary>
public sealed class ModelListResponse
{
    [JsonPropertyName("object")] public string Object => "list";
    [JsonPropertyName("data")] public required IReadOnlyList<ModelEntry> Data { get; init; }
}

/// <summary>Request to load/pull a model.</summary>
public sealed class ModelLoadRequest
{
    /// <summary>Local path or HuggingFace repo id.</summary>
    [JsonPropertyName("model")] public string Model { get; set; } = "";
}

/// <summary>One chat message, OpenAI wire shape.</summary>
public sealed class ChatMessageDto
{
    [JsonPropertyName("role")] public string Role { get; set; } = "user";
    [JsonPropertyName("content")] public string Content { get; set; } = "";
}

/// <summary>OpenAI chat-completions request. Sampling fields beyond temperature/top_p/max_tokens are
/// HartsyInference extensions (top_k, min_p, repetition_penalty, seed), matching what
/// <c>SamplingOptions</c> supports.</summary>
public sealed class ChatCompletionRequest
{
    [JsonPropertyName("model")] public string? Model { get; set; }
    [JsonPropertyName("messages")] public List<ChatMessageDto> Messages { get; set; } = [];
    [JsonPropertyName("stream")] public bool Stream { get; set; }
    [JsonPropertyName("max_tokens")] public int? MaxTokens { get; set; }
    [JsonPropertyName("temperature")] public float? Temperature { get; set; }
    [JsonPropertyName("top_p")] public float? TopP { get; set; }
    [JsonPropertyName("top_k")] public int? TopK { get; set; }
    [JsonPropertyName("min_p")] public float? MinP { get; set; }
    [JsonPropertyName("repetition_penalty")] public float? RepetitionPenalty { get; set; }
    /// <summary>OpenAI JSON-mode: <c>{"type":"json_object"}</c> forces every generated token to keep the
    /// output syntactically valid JSON (see <c>JsonGrammarStep</c>). Only <c>json_object</c> is supported —
    /// the richer <c>json_schema</c> mode (constraining to a specific schema, not just "valid JSON") is a
    /// separate, larger feature and is rejected with a clear error rather than silently ignored.</summary>
    [JsonPropertyName("response_format")] public ResponseFormatDto? ResponseFormat { get; set; }
    [JsonPropertyName("seed")] public ulong? Seed { get; set; }
}

/// <summary>OpenAI <c>response_format</c> object. Only <c>"type":"json_object"</c> is recognized.</summary>
public sealed class ResponseFormatDto
{
    [JsonPropertyName("type")] public string Type { get; set; } = "text";
}

/// <summary>Non-streaming chat-completion response, OpenAI wire shape.</summary>
public sealed class ChatCompletionResponse
{
    [JsonPropertyName("id")] public required string Id { get; init; }
    [JsonPropertyName("object")] public string Object => "chat.completion";
    [JsonPropertyName("created")] public required long Created { get; init; }
    [JsonPropertyName("model")] public required string Model { get; init; }
    [JsonPropertyName("choices")] public required IReadOnlyList<ChatCompletionChoice> Choices { get; init; }
    [JsonPropertyName("usage")] public required ChatUsage Usage { get; init; }
}

public sealed class ChatCompletionChoice
{
    [JsonPropertyName("index")] public int Index { get; init; }
    [JsonPropertyName("message")] public required ChatMessageDto Message { get; init; }
    [JsonPropertyName("finish_reason")] public required string FinishReason { get; init; }
}

public sealed class ChatUsage
{
    [JsonPropertyName("prompt_tokens")] public required int PromptTokens { get; init; }
    [JsonPropertyName("completion_tokens")] public required int CompletionTokens { get; init; }
    [JsonPropertyName("total_tokens")] public int TotalTokens => PromptTokens + CompletionTokens;
}

/// <summary>One SSE chunk for streaming chat completions (<c>chat.completion.chunk</c>), OpenAI wire shape.</summary>
public sealed class ChatCompletionChunk
{
    [JsonPropertyName("id")] public required string Id { get; init; }
    [JsonPropertyName("object")] public string Object => "chat.completion.chunk";
    [JsonPropertyName("created")] public required long Created { get; init; }
    [JsonPropertyName("model")] public required string Model { get; init; }
    [JsonPropertyName("choices")] public required IReadOnlyList<ChatCompletionChunkChoice> Choices { get; init; }
}

public sealed class ChatCompletionChunkChoice
{
    [JsonPropertyName("index")] public int Index { get; init; }
    [JsonPropertyName("delta")] public required ChatCompletionDelta Delta { get; init; }
    [JsonPropertyName("finish_reason")] public string? FinishReason { get; init; }
}

public sealed class ChatCompletionDelta
{
    [JsonPropertyName("role")] public string? Role { get; init; }
    [JsonPropertyName("content")] public string? Content { get; init; }
}

/// <summary>OpenAI error envelope.</summary>
public sealed class OpenAiError
{
    [JsonPropertyName("error")] public required OpenAiErrorBody Error { get; init; }

    public static OpenAiError Make(string message, string type) =>
        new() { Error = new OpenAiErrorBody { Message = message, Type = type } };
}

/// <summary>OpenAI error body.</summary>
public sealed class OpenAiErrorBody
{
    [JsonPropertyName("message")] public required string Message { get; init; }
    [JsonPropertyName("type")] public required string Type { get; init; }
}
