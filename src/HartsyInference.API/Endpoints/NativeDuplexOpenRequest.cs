using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.API.Endpoints;

/// <summary>The first text message of <c>/v1/native/duplex/ws</c>: which full-duplex model to open, how it behaves, and how the audio travels.</summary>
public sealed class NativeDuplexOpenRequest
{
    /// <summary>Catalog id with an optional precision (<c>haru</c>, <c>haru:q8</c>), local path, or HuggingFace repo id.</summary>
    public required string Model { get; set; }

    /// <summary>Explicit checkpoint path override; wins over catalog/HF resolution. Must be an absolute path.</summary>
    public string? ModelPath { get; set; }

    /// <summary>The character card, the voice reference and the sampling knobs.</summary>
    public DuplexSpeechRequest Request { get; set; } = new();

    /// <summary>Sample rate of the audio the client sends; 0 means the model's own (24 kHz for Haru). Other rates must divide by 50 (8000, 16000, 44100, 48000).</summary>
    public int InputSampleRate { get; set; }

    /// <summary>Frames of lead before the client starts playing (see <see cref="DuplexVoiceOptions.LeadFrames"/>); 0 to 8. Each is 133 ms of delay for Haru.</summary>
    public int LeadFrames { get; set; } = 2;

    /// <summary>Audio encoding in both directions: <c>pcm16</c> (little-endian signed 16-bit, the default) or <c>f32</c> (little-endian 32-bit float).</summary>
    public string Format { get; set; } = "pcm16";
}
