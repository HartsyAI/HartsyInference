namespace HartsyInference.Engine.Audio.Wake;

/// <summary>Configuration for the wake-word listener. Defaults suit a home LAN with a handful of satellites.</summary>
public sealed record WakeServiceOptions
{
    /// <summary>Interface to bind. Defaults to all interfaces because satellites connect from the LAN.</summary>
    public string BindAddress { get; init; } = "0.0.0.0";

    /// <summary>TCP port satellites connect to. 0 binds an ephemeral port, which tests use.</summary>
    public int Port { get; init; } = 10_800;

    /// <summary>Largest binary payload accepted on one frame. At 16 kHz mono this caps a frame at ~32 s of audio, far above the 20-40 ms frames a satellite should send; it exists to bound the damage from a corrupt or hostile header rather than to shape normal traffic.</summary>
    public int MaxPayloadBytes { get; init; } = 1 << 20;

    /// <summary>How often the server pings an idle connection. A satellite that lost its access point leaves a socket that still looks writable, so the absence of pongs is what actually surfaces the loss.</summary>
    public TimeSpan PingInterval { get; init; } = TimeSpan.FromSeconds(10);

    /// <summary>Whether to bind the raw TCP port. Turn it off to accept satellites only over a transport the host supplies (a WebSocket behind TLS, say) — then nothing is listening on the LAN at all.</summary>
    public bool EnableTcpListener { get; init; } = true;

    /// <summary>Shared secret a satellite must present in its <c>hello</c> frame. Null or empty disables the check, which is the sane default on a trusted LAN; set it before this endpoint is reachable from anywhere less trusted, because without it any device that can open the port can stream audio in and receive every detection — including the transcripts of what was said.</summary>
    public string? AuthToken { get; init; }

    /// <summary>Longest utterance captured around a detection, and the cap on how long end-of-speech will wait.</summary>
    public double UtteranceSeconds { get; init; } = 8.0;

    /// <summary>Silence, in milliseconds, that ends an utterance once <see cref="UseEndOfSpeech"/> is on and VAD
    /// weights are installed. 500 ms is about the shortest a person can pause mid-sentence without it reading as
    /// the end of one.</summary>
    public int EndOfSpeechSilenceMs { get; init; } = 500;

    /// <summary>Whether to end an utterance when the speaker stops rather than after a fixed wait.
    ///
    /// <para>Off, or with no VAD weights installed, transcription starts a fixed three seconds after the word
    /// fires and captures the preceding <see cref="UtteranceSeconds"/> — which truncates anyone whose question
    /// runs past that, and makes everyone else wait the full three seconds for a two-word command.</para></summary>
    public bool UseEndOfSpeech { get; init; } = true;

    /// <summary>Audio kept from before the detection, so the snapshot starts ahead of the wake word rather than
    /// clipping its first syllable.</summary>
    public double LeadInSeconds { get; init; } = 1.5;

    /// <summary>Directory holding wake assets (<c>vad/</c>, <c>backbone/</c>, <c>heads/</c>). Defaults to <c>{models}/audio/wake</c>.</summary>
    public string? ModelRoot { get; init; }

    /// <summary>Wake words to load, mapped to their per-word settings. Empty means every head found on disk, with default settings.</summary>
    public IReadOnlyDictionary<string, WakeWordConfig> Words { get; init; } = new Dictionary<string, WakeWordConfig>();

    /// <summary>Whether a detection should also transcribe the utterance that follows it.</summary>
    public bool TranscribeOnDetection { get; init; } = true;

    /// <summary>Model id used for that transcription.</summary>
    public string TranscribeModel { get; init; } = "whisper";

    /// <summary>Whether to identify who spoke and enforce per-word speaker restrictions. Requires CAM++ weights; when they are missing the service logs and runs ungated.</summary>
    public bool IdentifySpeakers { get; init; } = true;

    /// <summary>Whether to run RNNoise over each satellite's audio before scoring it. Requires denoiser weights under <c>{ModelRoot}/denoise</c>; when they are missing the service logs and runs without suppression rather than refusing to listen.
    ///
    /// <para>This is a property of the microphone and the room, not of a wake word, so it is deliberately global rather than per-word: one pipeline scores every head against one audio stream, and "which word's setting applies" would have no answer.</para>
    ///
    /// <para>Off by default. It costs real compute per stream and changes the audio every head sees, so it is an opt-in rather than something that silently starts happening on upgrade.</para></summary>
    public bool NoiseSuppression { get; init; }

    /// <summary>URLs that receive a JSON POST for every detection. This is how other services subscribe to one engine's wake events without being in-process — the same detection can drive several agents.</summary>
    public IReadOnlyList<string> Webhooks { get; init; } = [];

    /// <summary>Wraps the post-detection transcription call so the host can put it behind its own admission gate. The engine is not safely re-entrant per backend, so in the API server this routes through the same <c>InferenceQueue</c> every HTTP route uses — otherwise a detection could run Whisper on the shared backend while an image or video job is mid-generation. Null runs it directly, which is correct for a host that has no other traffic.</summary>
    public Func<Func<Task<string?>>, Task<string?>>? TranscribeGate { get; init; }
}

/// <summary>Per-word configuration. <see cref="Route"/> is opaque to the engine: it is echoed back on the detection event so a caller can send different words to different agents without the engine knowing what any of them mean.</summary>
public sealed record WakeWordConfig
{
    /// <summary>Head file name (without extension) under <c>heads/</c>. Defaults to the word's key.</summary>
    public string? Head { get; init; }

    /// <summary>Smoothed score at or above which the word fires.</summary>
    public float Threshold { get; init; } = 0.5f;

    /// <summary>Score steps averaged before thresholding.</summary>
    public int SmoothingWindow { get; init; } = 3;

    /// <summary>Silence enforced after a detection before this word can fire again.</summary>
    public double RefractorySeconds { get; init; } = 2.0;

    /// <summary>Caller-defined tag echoed on the detection event — an agent name, a room, a scene.</summary>
    public string? Route { get; init; }

    /// <summary>When set, the word only fires for this enrolled speaker.</summary>
    public string? RequiredSpeaker { get; init; }
}
