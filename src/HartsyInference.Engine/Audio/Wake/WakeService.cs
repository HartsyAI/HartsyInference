using System.Collections.Concurrent;
using System.Globalization;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Audio.Wake;

/// <summary>What a wake word firing produced: the word, the device, and — when transcription is enabled — the
/// command that followed it. <see cref="Route"/> is the caller's own tag from configuration, so one engine can
/// feed several agents without knowing anything about them.</summary>
public sealed record WakeEvent
{
    public required string DeviceId { get; init; }
    public required string Word { get; init; }
    public required float Score { get; init; }
    public string? Route { get; init; }
    public string? Transcript { get; init; }
    public string? Speaker { get; init; }
    public DateTimeOffset DetectedAtUtc { get; init; } = DateTimeOffset.UtcNow;
}

/// <summary>Always-on wake word detection over network satellites.
///
/// <para>Hosted by whatever process owns the engine — the standalone API server or the SwarmUI extension — so
/// devices connect to one endpoint regardless. Detections are delivered three ways: back to the originating
/// device over its own socket, to <see cref="Detected"/> for in-process subscribers, and (later) to registered
/// webhooks, which is what makes the same detection usable by several services at once.</para></summary>
public sealed class WakeService : IDisposable
{
    private readonly ConcurrentDictionary<string, WakeSession> _sessions = new();
    private readonly IInferenceEngine _engine;
    private readonly WakeServiceOptions _options;
    private readonly HttpClient _http = new() { Timeout = TimeSpan.FromSeconds(10) };
    private WakeModelSet? _models;
    private WakeListener? _listener;
    private WakeWorker? _worker;
    private WakeWordConfigStore? _configStore;
    private int _disposed;

    /// <summary>Raised on every detection, after transcription when it is enabled.</summary>
    public event Action<WakeEvent>? Detected;

    /// <summary>Devices with a session, connected or not — a session outlives its connection so a reconnecting
    /// device keeps its configuration.</summary>
    public IReadOnlyCollection<string> Devices => [.. _sessions.Keys];

    /// <summary>Wake words currently loaded.</summary>
    public IReadOnlyCollection<string> Words => _models?.Words ?? [];

    /// <summary>The bound listener port; 0 before <see cref="Start"/>.</summary>
    public int Port => _listener?.Port ?? 0;

    public WakeService(IInferenceEngine engine, WakeServiceOptions? options = null)
    {
        _engine = engine ?? throw new ArgumentNullException(nameof(engine));
        _options = options ?? new WakeServiceOptions();
    }

    /// <summary>Loads the models, starts the detection worker, then opens the listener. In that order, so a
    /// satellite is never accepted into a service that cannot yet score its audio.</summary>
    public void Start()
    {
        string root = _options.ModelRoot ?? Path.Combine(RepoPaths.ModelsRoot(), "audio", "wake");

        // Persisted per-word settings are the base; anything passed in options wins, so a host can override
        // without rewriting the file.
        _configStore = new WakeWordConfigStore(root);
        Dictionary<string, WakeWordConfig> words = new(_configStore.Load());
        foreach ((string name, WakeWordConfig config) in _options.Words) words[name] = config;

        _models = new WakeModelSet(root);
        _models.Load(words);

        _worker = new WakeWorker(_sessions, OnDetectionAsync);
        _worker.Start();

        _listener = new WakeListener(_sessions, CreateSession, _options);
        _listener.Start();
    }

    private WakeSession CreateSession(string deviceId) => new(deviceId, _models!.CreatePipeline());

    private async Task OnDetectionAsync(WakeSession session, WakeDetection detection)
    {
        string? transcript = null;
        if (_options.TranscribeOnDetection)
        {
            try
            {
                transcript = _options.TranscribeGate is null
                    ? await TranscribeUtteranceAsync(session).ConfigureAwait(false)
                    : await _options.TranscribeGate(() => TranscribeUtteranceAsync(session)).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                // A transcription failure must not swallow the detection itself — the caller still wants to
                // know the word fired.
                Logs.Error($"[Audio][Wake] Transcription failed after '{detection.Word}' on '{session.DeviceId}'.", ex);
            }
        }

        WakeWordConfig? config = _models?.ConfigFor(detection.Word);
        WakeEvent evt = new()
        {
            DeviceId = session.DeviceId,
            Word = detection.Word,
            Score = detection.Score,
            Route = config?.Route,
            Transcript = transcript,
        };

        await NotifyDeviceAsync(session, evt).ConfigureAwait(false);
        await PostWebhooksAsync(evt).ConfigureAwait(false);
        try { Detected?.Invoke(evt); }
        catch (Exception ex) { Logs.Error("[Audio][Wake] A detection subscriber threw.", ex); }
    }

    private async Task<string?> TranscribeUtteranceAsync(WakeSession session)
    {
        // Wait for the command that follows the wake word before transcribing; the detection fires as the
        // word ends, so the useful audio is still arriving.
        await Task.Delay(TimeSpan.FromSeconds(Math.Min(3.0, _options.UtteranceSeconds))).ConfigureAwait(false);

        float[] pcm = session.SnapshotRecent(_options.UtteranceSeconds);
        if (pcm.Length == 0) return null;

        // The wake path carries int16-scaled audio; the WAV writer wants ±1.
        float[] normalized = new float[pcm.Length];
        for (int i = 0; i < pcm.Length; i++) normalized[i] = pcm[i] / 32768f;

        using MemoryStream ms = new();
        WavFile.WriteMono16(ms, normalized, 16_000);

        ModelSpec spec = Registry.ModelResolver.Resolve(_options.TranscribeModel, null, Modality.Transcribe);
        TranscriptResult result = await _engine.Transcribe.RunAsync(spec, new AudioRequest
        {
            Audio = new AudioClip { Data = ms.ToArray(), Format = "wav" },
        }).ConfigureAwait(false);
        return result.Text;
    }

    /// <summary>Persists a word's settings so they survive a restart, and applies them to future sessions.</summary>
    public void ConfigureWord(string name, WakeWordConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _configStore?.Set(name, config);
        _models?.TryLoadHead(name, config);
    }

    /// <summary>Per-word settings currently in effect.</summary>
    public IReadOnlyDictionary<string, WakeWordConfig> WordSettings => _configStore?.Entries ?? new Dictionary<string, WakeWordConfig>();

    private async Task PostWebhooksAsync(WakeEvent evt)
    {
        if (_options.Webhooks.Count == 0) return;
        string payload = JsonSerializer.Serialize(evt);
        foreach (string url in _options.Webhooks)
        {
            try
            {
                using StringContent content = new(payload, Encoding.UTF8, "application/json");
                using HttpResponseMessage response = await _http.PostAsync(url, content).ConfigureAwait(false);
                if (!response.IsSuccessStatusCode)
                    Logs.Warning($"[Audio][Wake] Webhook {url} returned {(int)response.StatusCode}.");
            }
            catch (Exception ex)
            {
                // One unreachable subscriber must not stop the others or the detection itself.
                Logs.Warning($"[Audio][Wake] Webhook {url} failed: {ex.Message}");
            }
        }
    }

    private static async Task NotifyDeviceAsync(WakeSession session, WakeEvent evt)
    {
        WakeFrameCodec? codec = session.Codec;
        if (codec is null) return;
        try
        {
            string data = $"{{\"name\":{WakeFrameCodec.Escape(evt.Word)}," +
                $"\"score\":{evt.Score.ToString("F4", CultureInfo.InvariantCulture)}" +
                (evt.Route is null ? "" : $",\"route\":{WakeFrameCodec.Escape(evt.Route)}") +
                (evt.Transcript is null ? "" : $",\"transcript\":{WakeFrameCodec.Escape(evt.Transcript)}") +
                (evt.Speaker is null ? "" : $",\"speaker\":{WakeFrameCodec.Escape(evt.Speaker)}") + "}";
            await codec.WriteAsync("detection", data, CancellationToken.None).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            // The device may have dropped between detecting and reporting; its reconnect will re-register.
            Logs.Verbose($"[Audio][Wake] Could not deliver detection to '{session.DeviceId}': {ex.Message}");
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _listener?.Dispose();
        _worker?.Dispose();
        foreach (WakeSession session in _sessions.Values) session.Dispose();
        _sessions.Clear();
        _models?.Dispose();
        _http.Dispose();
    }
}
