using HartsyInference.Engine.Requests;

namespace HartsyInference.API.Endpoints;

/// <summary>Envelope for <c>/v1/native/speech</c>. <see cref="AudioClip.Data"/>/<see cref="SpeechRequest.Reference"/> bytes travel as base64 within the JSON body (System.Text.Json's built-in <c>byte[]</c> handling) — same envelope pattern as every other native route, no separate multipart plumbing.</summary>
public sealed class NativeSpeechRequest : NativeArtifactRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution. Must be an absolute path — a relative one resolves against the server process's working directory, not anything an HTTP client can know. Rarely needed: a plain catalog id in <c>model</c> resolves correctly on its own.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native speech request, unmodified.</summary>
    public required SpeechRequest Request { get; set; }
}

/// <summary>Envelope for <c>/v1/native/music</c>.</summary>
public sealed class NativeMusicRequest : NativeArtifactRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution. Must be an absolute path — a relative one resolves against the server process's working directory, not anything an HTTP client can know. Rarely needed: a plain catalog id in <c>model</c> resolves correctly on its own.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native music request, unmodified.</summary>
    public required MusicRequest Request { get; set; }
}

/// <summary>Envelope for <c>/v1/native/transcribe</c>.</summary>
public sealed class NativeTranscribeRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution. Must be an absolute path — a relative one resolves against the server process's working directory, not anything an HTTP client can know. Rarely needed: a plain catalog id in <c>model</c> resolves correctly on its own.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native transcription request, unmodified.</summary>
    public required AudioRequest Request { get; set; }
}

/// <summary>Envelope for <c>/v1/native/voice-convert</c>.</summary>
public sealed class NativeVoiceConvertRequest : NativeArtifactRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution. Must be an absolute path — a relative one resolves against the server process's working directory, not anything an HTTP client can know. Rarely needed: a plain catalog id in <c>model</c> resolves correctly on its own.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native voice-conversion request, unmodified.</summary>
    public required VoiceConversionRequest Request { get; set; }
}

/// <summary>Envelope for <c>/v1/native/fx/separate</c>.</summary>
public sealed class NativeFxSeparateRequest : NativeArtifactRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution. Must be an absolute path — a relative one resolves against the server process's working directory, not anything an HTTP client can know. Rarely needed: a plain catalog id in <c>model</c> resolves correctly on its own.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native fx-separate request, unmodified.</summary>
    public required FxSeparateRequest Request { get; set; }
}

/// <summary>Envelope for <c>/v1/native/fx/enhance</c>.</summary>
public sealed class NativeFxEnhanceRequest : NativeArtifactRequest
{
    /// <summary>Catalog id, local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution. Must be an absolute path — a relative one resolves against the server process's working directory, not anything an HTTP client can know. Rarely needed: a plain catalog id in <c>model</c> resolves correctly on its own.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The native fx-enhance request, unmodified.</summary>
    public required FxEnhanceRequest Request { get; set; }
}
