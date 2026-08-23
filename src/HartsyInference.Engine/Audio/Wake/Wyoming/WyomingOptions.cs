using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio.Wake.Wyoming;

/// <summary>Configuration for the Home Assistant / Wyoming compatibility endpoint.</summary>
public sealed record WyomingOptions
{
    /// <summary>Interface to bind. All interfaces, because Home Assistant dials in from the LAN.</summary>
    public string BindAddress { get; init; } = "0.0.0.0";

    /// <summary>TCP port Home Assistant connects to. Deliberately not 10800: the satellite listener owns that one and both run in the same process. 0 binds an ephemeral port, which tests use.</summary>
    public int Port { get; init; } = 10_600;

    /// <summary>Largest binary payload accepted on one frame.</summary>
    public int MaxPayloadBytes { get; init; } = 1 << 20;

    /// <summary>Program name shown in the manifest and matched by Wyoming's <c>select-program</c>.</summary>
    public string ProgramName { get; init; } = "hartsyinference";

    public string ProgramDescription { get; init; } = "HartsyInference — pure C#/.NET inference engine";

    public string? ProgramVersion { get; init; }

    public WyomingAttribution Attribution { get; init; } = WyomingAttribution.Engine;

    /// <summary>Speech-to-text models offered. Empty hides the <c>asr</c> service from Home Assistant.</summary>
    public IReadOnlyList<WyomingArtifact> AsrModels { get; init; } =
        [new WyomingArtifact { Name = "whisper", Description = "Whisper speech-to-text" }];

    /// <summary>Text-to-speech voices offered. Empty hides the <c>tts</c> service from Home Assistant.</summary>
    public IReadOnlyList<WyomingArtifact> TtsVoices { get; init; } =
        [new WyomingArtifact { Name = "kokoro", Description = "Kokoro-82M text-to-speech" }];

    /// <summary>Wake models offered. Only advertised when <see cref="WakeDetectorFactory"/> is also set — a service Home Assistant lists but that can never fire is worse than one it never sees.</summary>
    public IReadOnlyList<WyomingArtifact> WakeModels { get; init; } = [];

    /// <summary>Builds the detector for one wake connection from the words Home Assistant asked for (empty means every configured word). Null leaves wake unadvertised and every stream answered <c>not-detected</c>.</summary>
    public Func<IReadOnlyList<string>, IWyomingWakeDetector>? WakeDetectorFactory { get; init; }

    /// <summary>Wraps every transcription so the host can put it behind its own admission gate. The engine is not safely re-entrant per backend, so a host with other traffic must route this through the same <c>InferenceQueue</c> its HTTP routes use. Null runs it directly on the socket's task.</summary>
    public Func<Func<Task<TranscriptResult>>, Task<TranscriptResult>>? TranscribeGate { get; init; }

    /// <summary>The synthesis half of <see cref="TranscribeGate"/>; same reasoning, different result type.</summary>
    public Func<Func<Task<AudioResult>>, Task<AudioResult>>? SynthesizeGate { get; init; }

    /// <summary>Ceiling on one buffered utterance (default ~60 s at 16 kHz mono). A client that never sends <c>audio-stop</c> would otherwise grow this without limit.</summary>
    public int MaxUtteranceBytes { get; init; } = 16_000 * 2 * 60;

    /// <summary>Bytes per outgoing synthesized <c>audio-chunk</c>.</summary>
    public int SynthesisChunkBytes { get; init; } = 2048;

    /// <summary>Default language reported on a transcript when the request named none.</summary>
    public string DefaultLanguage { get; init; } = "en";
}
