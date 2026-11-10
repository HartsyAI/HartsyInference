using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Endpoints;

/// <summary>Body of <c>POST /v1/native/realtime/sessions</c>.</summary>
public sealed class NativeRealtimeRequest
{
    /// <summary>Catalog id, local path, or Hugging Face repo id of a streaming vision-language model (MOSS-VL-Realtime).</summary>
    public required string Model { get; set; }

    /// <summary>Explicit absolute checkpoint path; wins over catalog resolution.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The session settings; null opens with the model's defaults and an empty first prompt.</summary>
    public RealtimeVlmRequest? Request { get; set; }
}

/// <summary>Body of <c>POST /v1/native/realtime/sessions/{id}/frame</c>.</summary>
public sealed class RealtimeFrameRequest
{
    /// <summary>The frame as base64 PNG (a <c>data:image/png;base64,</c> URI is accepted).</summary>
    public required string Image { get; set; }

    /// <summary>The frame's time in the stream, in seconds; never earlier than the previous frame's.</summary>
    public required double Timestamp { get; set; }
}

/// <summary>Body of <c>POST /v1/native/realtime/sessions/{id}/prompt</c>.</summary>
public sealed class RealtimePromptRequest
{
    /// <summary>The user prompt, spliced into the stream after the model's next token.</summary>
    public required string Prompt { get; set; }
}
