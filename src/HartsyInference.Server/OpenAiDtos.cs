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
