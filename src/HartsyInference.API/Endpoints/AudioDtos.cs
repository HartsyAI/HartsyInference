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

/// <summary>Envelope for <c>/v1/native/duplex/respond</c>: a recording of the user played to a full-duplex speech-to-speech model. The
/// recording and the voice reference travel as base64 inside the JSON body.</summary>
public sealed class NativeDuplexRespondRequest : NativeArtifactRequest
{
    /// <summary>Catalog id with an optional precision (<c>haru</c>, <c>haru:q4</c>), local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution. Must be an absolute path.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The character card, the voice reference and the sampling knobs.</summary>
    public DuplexSpeechRequest Request { get; set; } = new();

    /// <summary>The user's recording, any rate, mixed down to mono.</summary>
    public required AudioClip User { get; set; }

    /// <summary>Frames of silence after the recording, so the model can finish speaking (3200 samples each, 133 ms).</summary>
    public int TailFrames { get; set; }

    /// <summary>Return the conversation as a stereo WAV, the user on the left and the model on the right, instead of the model alone.</summary>
    public bool Stereo { get; set; }
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

/// <summary>Envelope for <c>/v1/native/music/score</c> and <c>/v1/native/music/budget</c>. Neither produces
/// audio, so unlike <see cref="NativeMusicRequest"/> there is nothing to persist.</summary>
public sealed class NativeMusicScoreRequest
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
