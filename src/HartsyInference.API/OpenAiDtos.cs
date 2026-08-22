using System.Text.Json;
using System.Text.Json.Serialization;

namespace HartsyInference.API;

/// <summary>OpenAI-compatible request/response DTOs. Field names use snake_case via attributes to match the OpenAI Images / error wire format so the official client SDKs interoperate.</summary>
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

    /// <summary>Denoise steps (HartsyInference extension); omitted uses the model family's recommended count.</summary>
    [JsonPropertyName("steps")] public int? Steps { get; set; }

    /// <summary>Guidance scale (HartsyInference extension); omitted uses the model family's recommended scale.</summary>
    [JsonPropertyName("cfg_scale")] public float? CfgScale { get; set; }

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

    /// <summary>Unix seconds. OpenAI's schema expects a real per-model creation date; this catalog doesn't track one, so it reports the server process's start time instead (same value for every entry, every request) — cosmetic schema-shape compliance, not a real timestamp claim.</summary>
    [JsonPropertyName("created")] public required long Created { get; init; }
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

    /// <summary>Null when this message carries only <see cref="ToolCalls"/> (OpenAI sets content null on an assistant turn whose <c>finish_reason</c> is <c>tool_calls</c>).</summary>
    [JsonPropertyName("content")] public string? Content { get; set; } = "";

    /// <summary>Populated on an assistant message that invoked one or more tools.</summary>
    [JsonPropertyName("tool_calls")] public List<ChatToolCallDto>? ToolCalls { get; set; }
}

/// <summary>One tool definition a client may offer the model, OpenAI wire shape. Only <c>"type":"function"</c> is recognized (OpenAI has no other tool type today).</summary>
public sealed class ChatToolDto
{
    [JsonPropertyName("type")] public string Type { get; set; } = "function";
    [JsonPropertyName("function")] public required ChatToolFunctionDto Function { get; set; }
}

public sealed class ChatToolFunctionDto
{
    [JsonPropertyName("name")] public required string Name { get; set; }
    [JsonPropertyName("description")] public string Description { get; set; } = "";

    /// <summary>Raw JSON-schema object for the tool's arguments — kept as a <see cref="JsonElement"/> and re-serialized to a string for the native <c>ToolDefinition.JsonSchema</c>, not parsed/validated here.</summary>
    [JsonPropertyName("parameters")] public JsonElement Parameters { get; set; }
}

/// <summary>A tool call the model emitted, OpenAI wire shape.</summary>
public sealed class ChatToolCallDto
{
    [JsonPropertyName("id")] public required string Id { get; init; }
    [JsonPropertyName("type")] public string Type => "function";
    [JsonPropertyName("index")] public int? Index { get; init; }
    [JsonPropertyName("function")] public required ChatToolCallFunctionDto Function { get; init; }
}

public sealed class ChatToolCallFunctionDto
{
    [JsonPropertyName("name")] public string? Name { get; init; }
    [JsonPropertyName("arguments")] public string? Arguments { get; init; }
}

/// <summary>OpenAI chat-completions request. Sampling fields beyond temperature/top_p/max_tokens are HartsyInference extensions (top_k, min_p, repetition_penalty, seed), matching what <c>SamplingOptions</c> supports.</summary>
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
    /// <summary>OpenAI JSON-mode: <c>{"type":"json_object"}</c> forces every generated token to keep the output syntactically valid JSON (see <c>JsonGrammarStep</c>). Only <c>json_object</c> is supported — the richer <c>json_schema</c> mode (constraining to a specific schema, not just "valid JSON") is a separate, larger feature and is rejected with a clear error rather than silently ignored.</summary>
    [JsonPropertyName("response_format")] public ResponseFormatDto? ResponseFormat { get; set; }
    [JsonPropertyName("seed")] public ulong? Seed { get; set; }

    /// <summary>Tools the model may call. Passed through to the native <c>TextRequest.Tools</c> unmodified (name/description/JSON-schema) — the native tool-calling path is fully built; this is DTO plumbing.</summary>
    [JsonPropertyName("tools")] public List<ChatToolDto>? Tools { get; set; }

    /// <summary>OpenAI's <c>tool_choice</c>: either a bare string (<c>"none"</c>/<c>"auto"</c>/<c>"required"</c>) or <c>{"type":"function","function":{"name":...}}</c> to force one specific tool. Kept as a raw <see cref="JsonElement"/> and parsed in <c>CompatEndpoints.ToTextRequest</c> rather than a custom converter, since it's one call site. <c>"required"</c> (call SOME tool, model's choice) has no native equivalent — <c>ForceToolId</c> forces one *specific* tool — so it best-effort maps to the same behavior as <c>"auto"</c> rather than guessing which tool to force.</summary>
    [JsonPropertyName("tool_choice")] public JsonElement? ToolChoice { get; set; }
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
    [JsonPropertyName("tool_calls")] public List<ChatToolCallDto>? ToolCalls { get; init; }
}

/// <summary>OpenAI text-to-speech request. <c>voice</c> is passed straight through as the engine's built-in voice name (e.g. a Kokoro voice pack) rather than mapped from OpenAI's fixed voice enum (alloy/echo/fable/onyx/nova/shimmer) — those names don't correspond to anything this engine ships, so pass a real voice name from the target model's own catalog instead. <c>response_format</c> only accepts <c>"wav"</c> (the default, also used when omitted): <c>AudioResult.Data</c> is always a pre-encoded WAV container — there's no mp3/opus/aac encoder to produce anything else, so an unsupported format is rejected with a clear 400 rather than silently returning WAV bytes mislabeled as something else.</summary>
public sealed class SpeechGenerationRequest
{
    [JsonPropertyName("model")] public string? Model { get; set; }
    [JsonPropertyName("input")] public string Input { get; set; } = "";
    [JsonPropertyName("voice")] public string? Voice { get; set; }
    [JsonPropertyName("speed")] public double? Speed { get; set; }
    [JsonPropertyName("response_format")] public string ResponseFormat { get; set; } = "wav";
}

/// <summary>OpenAI embeddings request. <c>input</c> is kept as a raw <see cref="JsonElement"/> since OpenAI accepts either a single string or an array of strings — parsed in <c>CompatEndpoints</c> rather than a custom converter, since it's one call site. <c>encoding_format</c> only supports <c>"float"</c> (the default); OpenAI's <c>"base64"</c> option isn't implemented, so it's rejected rather than silently returning float arrays under a base64-shaped request. A <c>dimensions</c> request that doesn't match the model's real output width is rejected too — no truncation, since Matryoshka-style truncate-and-renormalize correctness hasn't been verified for the embedding models this engine actually ships.</summary>
public sealed class EmbeddingsRequest
{
    [JsonPropertyName("model")] public string? Model { get; set; }
    [JsonPropertyName("input")] public JsonElement Input { get; set; }
    [JsonPropertyName("encoding_format")] public string EncodingFormat { get; set; } = "float";
    [JsonPropertyName("dimensions")] public int? Dimensions { get; set; }
}

/// <summary>OpenAI embeddings response envelope.</summary>
public sealed class EmbeddingsResponse
{
    [JsonPropertyName("object")] public string Object => "list";
    [JsonPropertyName("data")] public required IReadOnlyList<EmbeddingDataDto> Data { get; init; }
    [JsonPropertyName("model")] public required string Model { get; init; }
    [JsonPropertyName("usage")] public required EmbeddingsUsage Usage { get; init; }
}

/// <summary>One embedding vector, OpenAI wire shape.</summary>
public sealed class EmbeddingDataDto
{
    [JsonPropertyName("object")] public string Object => "embedding";
    [JsonPropertyName("embedding")] public required float[] Embedding { get; init; }
    [JsonPropertyName("index")] public required int Index { get; init; }
}

/// <summary>Real token counts from the same tokenize pass that produced the vectors — not an estimate.</summary>
public sealed class EmbeddingsUsage
{
    [JsonPropertyName("prompt_tokens")] public required int PromptTokens { get; init; }
    [JsonPropertyName("total_tokens")] public required int TotalTokens { get; init; }
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
