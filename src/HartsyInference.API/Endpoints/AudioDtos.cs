using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Endpoints;

/// <summary>Envelope for <c>/v1/native/speech</c>. <see cref="AudioClip.Data"/>/<see cref="SpeechRequest.Reference"/>
/// bytes travel as base64 within the JSON body (System.Text.Json's built-in <c>byte[]</c> handling) — same envelope
/// pattern as every other native route, no separate multipart plumbing.</summary>
public sealed class NativeSpeechRequest
{
    public required string Model { get; set; }
    public string? ModelPath { get; set; }
    public required SpeechRequest Request { get; set; }
}

/// <summary>Envelope for <c>/v1/native/transcribe</c>.</summary>
public sealed class NativeTranscribeRequest
{
    public required string Model { get; set; }
    public string? ModelPath { get; set; }
    public required AudioRequest Request { get; set; }
}

/// <summary>Envelope for <c>/v1/native/voice-convert</c>.</summary>
public sealed class NativeVoiceConvertRequest
{
    public required string Model { get; set; }
    public string? ModelPath { get; set; }
    public required VoiceConversionRequest Request { get; set; }
}

/// <summary>Envelope for <c>/v1/native/fx/separate</c>.</summary>
public sealed class NativeFxSeparateRequest
{
    public required string Model { get; set; }
    public string? ModelPath { get; set; }
    public required FxSeparateRequest Request { get; set; }
}

/// <summary>Envelope for <c>/v1/native/fx/enhance</c>.</summary>
public sealed class NativeFxEnhanceRequest
{
    public required string Model { get; set; }
    public string? ModelPath { get; set; }
    public required FxEnhanceRequest Request { get; set; }
}
