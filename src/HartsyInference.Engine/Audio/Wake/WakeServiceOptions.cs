namespace HartsyInference.Engine.Audio.Wake;

/// <summary>Configuration for the wake-word listener. Defaults suit a home LAN with a handful of satellites.</summary>
public sealed record WakeServiceOptions
{
    /// <summary>Interface to bind. Defaults to all interfaces because satellites connect from the LAN.</summary>
    public string BindAddress { get; init; } = "0.0.0.0";

    /// <summary>TCP port satellites connect to. 0 binds an ephemeral port, which tests use.</summary>
    public int Port { get; init; } = 10_800;

    /// <summary>Largest binary payload accepted on one frame. At 16 kHz mono this caps a frame at ~32 s of
    /// audio, far above the 20-40 ms frames a satellite should send; it exists to bound the damage from a
    /// corrupt or hostile header rather than to shape normal traffic.</summary>
    public int MaxPayloadBytes { get; init; } = 1 << 20;

    /// <summary>How often the server pings an idle connection. A satellite that lost its access point leaves a
    /// socket that still looks writable, so the absence of pongs is what actually surfaces the loss.</summary>
    public TimeSpan PingInterval { get; init; } = TimeSpan.FromSeconds(10);

    /// <summary>Audio captured around a detection and handed to transcription.</summary>
    public double UtteranceSeconds { get; init; } = 8.0;

    /// <summary>Directory holding wake assets (<c>vad/</c>, <c>backbone/</c>, <c>heads/</c>). Defaults to
    /// <c>{models}/audio/wake</c>.</summary>
    public string? ModelRoot { get; init; }

    /// <summary>Wake words to load, mapped to their per-word settings. Empty means every head found on disk,
    /// with default settings.</summary>
    public IReadOnlyDictionary<string, WakeWordConfig> Words { get; init; } = new Dictionary<string, WakeWordConfig>();

    /// <summary>Whether a detection should also transcribe the utterance that follows it.</summary>
    public bool TranscribeOnDetection { get; init; } = true;

    /// <summary>Model id used for that transcription.</summary>
    public string TranscribeModel { get; init; } = "whisper";

    /// <summary>Wraps the post-detection transcription call so the host can put it behind its own admission
    /// gate. The engine is not safely re-entrant per backend, so in the API server this routes through the
    /// same <c>InferenceQueue</c> every HTTP route uses — otherwise a detection could run Whisper on the
    /// shared backend while an image or video job is mid-generation. Null runs it directly, which is correct
    /// for a host that has no other traffic.</summary>
    public Func<Func<Task<string?>>, Task<string?>>? TranscribeGate { get; init; }
}

/// <summary>Per-word configuration. <see cref="Route"/> is opaque to the engine: it is echoed back on the
/// detection event so a caller can send different words to different agents without the engine knowing what
/// any of them mean.</summary>
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
