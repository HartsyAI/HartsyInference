namespace HartsyInference.Engine.Requests;

/// <summary>Opens a live vision-language session: frames and prompts stream in, the model's reply streams out.</summary>
public sealed record RealtimeVlmRequest
{
    /// <summary>The device to run on; null uses the engine's primary device.</summary>
    public string? Device { get; init; }

    /// <summary>The system prompt; null uses the model's own live-session prompt.</summary>
    public string? SystemPrompt { get; init; }

    /// <summary>The first user prompt, which opens the chat before any frame arrives. May be empty.</summary>
    public string InitialPrompt { get; init; } = string.Empty;

    /// <summary>Frames waiting for the next cycle; when full, the oldest is dropped.</summary>
    public int FrameQueueSize { get; init; } = 256;

    /// <summary>The most tokens decoded per second; null leaves the stream unpaced (the reference's default).</summary>
    public double? MaxTokensPerSecond { get; init; }

    /// <summary>The most frames one cycle takes from the queue; null takes them all.</summary>
    public int? MaxFramesPerCycle { get; init; }

    /// <summary>The most vision tokens the session holds before it refuses more frames; null uses the model default.</summary>
    public int? MaxVisionTokens { get; init; }

    /// <summary>Sampling temperature; null (with the other sampling fields null) decodes greedily, as the reference does.</summary>
    public float? Temperature { get; init; }

    /// <summary>Nucleus sampling mass; null keeps every token.</summary>
    public float? TopP { get; init; }

    /// <summary>Sampling seed.</summary>
    public ulong? Seed { get; init; }

    /// <summary>Low-VRAM on-the-fly quant to load the model at, as for <see cref="TextRequest.LowVramQuant"/>.</summary>
    public string? LowVramQuant { get; init; }
}

/// <summary>What a live session reports.</summary>
public enum RealtimeVlmChunkKind
{
    /// <summary>A pushed prompt entered the stream; <see cref="RealtimeVlmChunk.Text"/> is the prompt.</summary>
    RoundStart,

    /// <summary>Reply text.</summary>
    Text,

    /// <summary>The model stayed silent.</summary>
    Silence,

    /// <summary>The model emitted another control token; <see cref="RealtimeVlmChunk.Text"/> is its literal (MOSS-VL: <c>&lt;|response|&gt;</c>).</summary>
    Control,

    /// <summary>Frames were dropped from a full queue; <see cref="RealtimeVlmChunk.Count"/> says how many.</summary>
    FramesDropped,

    /// <summary>The session failed and ended; <see cref="RealtimeVlmChunk.Text"/> is the reason.</summary>
    Error,
}

/// <summary>One event of a live session.</summary>
public sealed record RealtimeVlmChunk(RealtimeVlmChunkKind Kind, string Text = "", int Count = 0);

/// <summary>Counters of a live session.</summary>
public sealed record RealtimeVlmSessionStats(int TextTokens, int VisionTokens, long GeneratedTokens, long FramesProcessed, long FramesDropped);

/// <summary>A running live vision-language session. It holds its device's text slot until disposed: other text requests and unloads on
/// that device wait, while work on the device's other services is gated per forward.</summary>
// TODO(websocket): the session contract is transport-neutral; a duplex transport can drive PushFrame/PushPrompt and StreamAsync directly.
public interface IRealtimeVlmSession : IAsyncDisposable
{
    /// <summary>Queues a frame captured at <paramref name="timestampSeconds"/> (non-decreasing across calls). Returns true when the queue
    /// was full and its oldest frame was dropped.</summary>
    bool PushFrame(ImageData frame, double timestampSeconds);

    /// <summary>Queues a user prompt, spliced in after the next token.</summary>
    void PushPrompt(string prompt);

    /// <summary>The session's events until it ends. One reader at a time.</summary>
    IAsyncEnumerable<RealtimeVlmChunk> StreamAsync(CancellationToken cancel = default);

    /// <summary>The session's counters.</summary>
    RealtimeVlmSessionStats Stats { get; }
}
