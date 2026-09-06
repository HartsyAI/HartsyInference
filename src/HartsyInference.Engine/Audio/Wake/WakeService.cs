using System.Collections.Concurrent;
using System.Globalization;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Wake;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Cpu;
using HartsyInference.Engine.Audio.Wake.Speakers;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Audio.Wake;

/// <summary>What a wake word firing produced: the word, the device, and — when transcription is enabled — the command that followed it. <see cref="Route"/> is the caller's own tag from configuration, so one engine can feed several agents without knowing anything about them.</summary>
public sealed record WakeEvent
{
    public required string DeviceId { get; init; }
    public required string Word { get; init; }
    public required float Score { get; init; }
    public string? Route { get; init; }
    public string? Transcript { get; init; }

    /// <summary>The transcript with the wake phrase removed — what the user actually asked. Null when there was
    /// no transcript. See <see cref="WakePhrase"/> for why both are reported.</summary>
    public string? Command { get; init; }

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
    private SpeakerVerifier? _speakers;
    private IBackend? _speakerBackend;
    private int _disposed;

    /// <summary>Raised on every detection, after transcription when it is enabled.</summary>
    public event Action<WakeEvent>? Detected;

    /// <summary>Devices with a session, connected or not — a session outlives its connection so a reconnecting device keeps its configuration.</summary>
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

    /// <summary>Loads the models, starts the detection worker, then opens the listener. In that order, so a satellite is never accepted into a service that cannot yet score its audio.</summary>
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

        if (_options.NoiseSuppression) _models.LoadDenoiser();

        if (_options.IdentifySpeakers && SpeakerVerifier.IsAvailable)
        {
            try
            {
                _speakers = SpeakerVerifier.Load();
                // Its own backend: speaker embedding is burst work but must not contend with the shared
                // inference backend that HTTP generation runs on.
                _speakerBackend = new CpuBackend();
                Logs.Info($"[Audio][Wake] Speaker identification enabled ({_speakers.Store.Count} enrolled).");
            }
            catch (Exception ex)
            {
                Logs.Warning($"[Audio][Wake] Speaker identification unavailable, continuing ungated: {ex.Message}");
            }
        }

        _worker = new WakeWorker(_sessions, OnDetectionAsync);
        _worker.Start();

        // The listener object is always created — it owns the protocol loop that ServeConnectionAsync reuses —
        // but the TCP socket is only bound when asked for.
        _listener = new WakeListener(_sessions, CreateSession, _options);
        if (_options.EnableTcpListener)
        {
            _listener.Start();
        }
        else
        {
            Logs.Info("[Audio][Wake] TCP listener disabled; satellites must connect through a host-supplied transport.");
        }
    }

    private WakeSession CreateSession(string deviceId) => new(deviceId, _models!.CreatePipeline(), _models.CreateDenoiser());

    private async Task OnDetectionAsync(WakeSession session, WakeDetection detection)
    {
        WakeWordConfig? config = _models?.ConfigFor(detection.Word);

        // Tell the device the word fired BEFORE doing anything slow. Transcription has to wait for the command
        // that follows the wake word and may load a model first, so folding it into one event left a satellite
        // silent for seconds after the user spoke — long enough to look broken. The device acknowledges now
        // (light, chime) and acts on the transcript when it arrives.
        await NotifyDeviceAsync(session, "detection", new WakeEvent
        {
            DeviceId = session.DeviceId,
            Word = detection.Word,
            Score = detection.Score,
            Route = config?.Route,
        }).ConfigureAwait(false);

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

        // Speaker identification runs here rather than before transcription because it scores the wake word
        // together with the command that follows it, and that audio has only just arrived.
        string? speaker = IdentifySpeaker(session, config, out bool speakerAllowed);
        if (!speakerAllowed)
        {
            Logs.Info($"[Audio][Wake] '{detection.Word}' on '{session.DeviceId}' ignored: requires speaker '{config?.RequiredSpeaker}', heard '{speaker ?? "unknown"}'.");
            await NotifyDeviceAsync(session, "detection-rejected", new WakeEvent
            {
                DeviceId = session.DeviceId,
                Word = detection.Word,
                Score = detection.Score,
                Speaker = speaker,
            }).ConfigureAwait(false);
            return;
        }

        WakeEvent evt = new()
        {
            DeviceId = session.DeviceId,
            Word = detection.Word,
            Score = detection.Score,
            Route = config?.Route,
            Transcript = transcript,
            Command = transcript is null ? null : WakePhrase.Strip(transcript, detection.Word),
            Speaker = speaker,
        };

        // The completed event: the device gets the transcript it was waiting for, and in-process subscribers
        // plus webhooks see one event carrying everything.
        await NotifyDeviceAsync(session, "transcript", evt).ConfigureAwait(false);
        await PostWebhooksAsync(evt).ConfigureAwait(false);
        try { Detected?.Invoke(evt); }
        catch (Exception ex) { Logs.Error("[Audio][Wake] A detection subscriber threw.", ex); }
    }

    /// <summary>Captures the utterance around a detection and transcribes it.</summary>
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

    /// <summary>Runs the satellite protocol over a caller-supplied stream, joining the same session machinery the
    /// TCP listener uses.
    ///
    /// <para>This exists so a host can accept satellites over a transport the engine does not own — a WebSocket
    /// behind TLS, for instance, which is what makes the listener reachable through an HTTP reverse proxy or
    /// tunnel that cannot carry raw TCP.</para></summary>
    public Task ServeConnectionAsync(Stream stream, string remoteLabel, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (_listener is null) throw new InvalidOperationException("Start the wake service before serving a connection.");
        return _listener.ServeStreamAsync(stream, remoteLabel, cancel);
    }

    /// <summary>Identifies who spoke, and decides whether a word restricted to one speaker may fire.
    ///
    /// <para>Scores the wake word together with the command that followed it: a wake phrase alone is around a
    /// second, and text-independent verification degrades badly at that length.</para>
    ///
    /// <para>When identification is unavailable, a <c>RequiredSpeaker</c> restriction cannot be honoured. It
    /// fails CLOSED — the word does not fire — because silently ignoring the restriction would be worse than
    /// missing a trigger.</para></summary>
    private string? IdentifySpeaker(WakeSession session, WakeWordConfig? config, out bool allowed)
    {
        allowed = true;
        bool restricted = !string.IsNullOrWhiteSpace(config?.RequiredSpeaker);
        if (_speakers is null || _speakerBackend is null)
        {
            if (restricted) allowed = false;
            return null;
        }

        try
        {
            float[] utterance = session.SnapshotRecent(_options.UtteranceSeconds);
            SpeakerMatch match = _speakers.Identify(_speakerBackend, utterance);
            if (restricted) allowed = match.Satisfies(config!.RequiredSpeaker);
            return match.IdentifiedName;
        }
        catch (Exception ex)
        {
            Logs.Warning($"[Audio][Wake] Speaker identification failed for '{session.DeviceId}': {ex.Message}");
            if (restricted) allowed = false;
            return null;
        }
    }

    /// <summary>Persists a word's settings and rolls them out to every live session.
    ///
    /// <para>The change is queued rather than applied here: pipelines are single-threaded and owned by the wake
    /// worker, so editing one from a request thread would race a push already in flight. The worker picks the
    /// change up between drains.</para></summary>
    public void ConfigureWord(string name, WakeWordConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _configStore?.Set(name, config);
        if (_models is null || !_models.TryLoadHead(name, config)) return;

        (WakeHead Head, WakeWordConfig Config)? entry = _models.Entry(name);
        if (entry is null) return;
        foreach (WakeSession session in _sessions.Values)
            session.PendingWords.Enqueue((name, entry.Value.Head, WakeModelSet.Settings(entry.Value.Config)));
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

    private static async Task NotifyDeviceAsync(WakeSession session, string type, WakeEvent evt)
    {
        WakeFrameCodec? codec = session.Codec;
        if (codec is null) return;
        try
        {
            string data = $"{{\"name\":{WakeFrameCodec.Escape(evt.Word)}," +
                $"\"score\":{evt.Score.ToString("F4", CultureInfo.InvariantCulture)}" +
                (evt.Route is null ? "" : $",\"route\":{WakeFrameCodec.Escape(evt.Route)}") +
                (evt.Transcript is null ? "" : $",\"transcript\":{WakeFrameCodec.Escape(evt.Transcript)}") +
                (evt.Command is null ? "" : $",\"command\":{WakeFrameCodec.Escape(evt.Command)}") +
                (evt.Speaker is null ? "" : $",\"speaker\":{WakeFrameCodec.Escape(evt.Speaker)}") + "}";
            await codec.WriteAsync(type, data, CancellationToken.None).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            // The device may have dropped between detecting and reporting; its reconnect will re-register.
            Logs.Verbose($"[Audio][Wake] Could not deliver '{type}' to '{session.DeviceId}': {ex.Message}");
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
        _speakers?.Dispose();
        _speakerBackend?.Dispose();
        _http.Dispose();
    }
}
