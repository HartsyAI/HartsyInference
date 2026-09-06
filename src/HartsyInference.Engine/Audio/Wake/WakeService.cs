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
        if (_options.UseEndOfSpeech) _models.LoadVad();

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

    private WakeSession CreateSession(string deviceId) =>
        new(deviceId, _models!.CreatePipeline(), _models.CreateDenoiser(),
            _options.UseEndOfSpeech ? _models.CreateVad(_options.EndOfSpeechSilenceMs) : null);

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
                // Two states, not one, because they are two different waits and the device should show them
                // differently: the first ends when the speaker stops talking, the second when a model finishes.
                transcript = _options.TranscribeGate is null
                    ? await TranscribeUtteranceAsync(session).ConfigureAwait(false)
                    : await _options.TranscribeGate(() => TranscribeUtteranceAsync(session)).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                // A transcription failure must not swallow the detection itself — the caller still wants to
                // know the word fired.
                Logs.Error($"[Audio][Wake] Transcription failed after '{detection.Word}' on '{session.DeviceId}'.", ex);
                await SendStatusAsync(session, WakeStatus.Error, "transcription failed").ConfigureAwait(false);
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
            await SendStatusAsync(session, WakeStatus.Done).ConfigureAwait(false);
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
        // Wait for the command that follows the wake word before transcribing; the detection fires as the word
        // ends, so the useful audio is still arriving. How long to wait is the whole question: too short and a
        // long question is cut off mid-word, too long and every short command pays for it.
        double captureSeconds = await WaitForUtteranceEndAsync(session).ConfigureAwait(false);
        // The speaker can stop now, and on a device with a light that is worth saying: end-of-speech means
        // the useful audio is already in hand, and everything after this is the server's problem.
        await SendStatusAsync(session, WakeStatus.Captured).ConfigureAwait(false);

        float[] pcm = session.SnapshotRecent(captureSeconds);
        if (pcm.Length == 0) return null;

        // The wake path carries int16-scaled audio; the WAV writer wants ±1.
        float[] normalized = new float[pcm.Length];
        for (int i = 0; i < pcm.Length; i++) normalized[i] = pcm[i] / 32768f;

        using MemoryStream ms = new();
        WavFile.WriteMono16(ms, normalized, 16_000);

        await SendStatusAsync(session, WakeStatus.Transcribing).ConfigureAwait(false);
        ModelSpec spec = Registry.ModelResolver.Resolve(_options.TranscribeModel, null, Modality.Transcribe);
        TranscriptResult result = await _engine.Transcribe.RunAsync(spec, new AudioRequest
        {
            Audio = new AudioClip { Data = ms.ToArray(), Format = "wav" },
        }).ConfigureAwait(false);
        return result.Text;
    }

    /// <summary>Waits for the speaker to finish, and returns how many seconds of history to transcribe.
    ///
    /// <para>With an end-of-speech detector, this returns as soon as the room has been quiet for
    /// <see cref="WakeServiceOptions.EndOfSpeechSilenceMs"/> — so a two-word command is transcribed almost
    /// immediately and a long question is allowed to finish. Without one it falls back to the fixed three-second
    /// wait, which is what shipped before and which truncates anything longer than that.</para>
    ///
    /// <para>The returned span covers the wait plus <see cref="WakeServiceOptions.LeadInSeconds"/> of audio from
    /// before the word fired, because the detection lands as the phrase ends and transcription of a command
    /// missing its opening syllable is worse than transcription of a little extra silence. It is capped at
    /// <see cref="WakeServiceOptions.UtteranceSeconds"/>, which is also the cap on the wait itself: someone who
    /// never stops talking still gets an answer.</para></summary>
    private async Task<double> WaitForUtteranceEndAsync(WakeSession session)
    {
        double cap = _options.UtteranceSeconds;
        if (session.Vad is null)
        {
            await Task.Delay(TimeSpan.FromSeconds(Math.Min(3.0, cap))).ConfigureAwait(false);
            return cap;
        }

        // The wake phrase itself is speech, so the detector has just seen some; starting the silence clock now
        // rather than trusting a stale value keeps a device that was already quiet from ending instantly.
        session.LastSpeechTicks = Environment.TickCount64;

        long start = Environment.TickCount64;
        // The cap is on how long someone may keep talking after the word fires, not on that plus the lead-in
        // already sitting in the buffer: subtracting the lead-in here would silently shorten the longest
        // question the device accepts, which is the truncation this whole path exists to remove.
        long maxWaitMs = (long)(cap * 1000);
        int silenceMs = Math.Max(0, _options.EndOfSpeechSilenceMs);
        while (true)
        {
            await Task.Delay(PollIntervalMs).ConfigureAwait(false);
            long now = Environment.TickCount64;
            long waited = now - start;
            if (now - session.LastSpeechTicks >= silenceMs)
            {
                double spoken = (waited + silenceMs) / 1000.0;
                return Math.Min(cap, spoken + _options.LeadInSeconds);
            }
            if (waited >= maxWaitMs)
            {
                Logs.Verbose($"[Audio][Wake] '{session.DeviceId}' still speaking after {cap:F1}s; transcribing what there is.");
                return cap;
            }
        }
    }

    /// <summary>How often the end-of-speech wait re-checks. Below the VAD's own 32 ms window there is nothing
    /// new to see, and above ~100 ms the poll itself would show up in the latency it is measuring.</summary>
    private const int PollIntervalMs = 50;

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

    /// <summary>Tells one satellite what is happening, so its light can say so.
    ///
    /// <para>Public because the states that matter most — <see cref="WakeStatus.Thinking"/> and
    /// <see cref="WakeStatus.Speaking"/> — belong to whoever owns the turn after the transcript leaves here,
    /// and that is not this class. A device that is not connected, or that never existed, is not an error:
    /// this is advisory, and a turn must not fail because a light could not be updated.</para></summary>
    /// <param name="deviceId">The satellite to tell. Unknown ids are ignored.</param>
    /// <param name="state">One of the <see cref="WakeStatus"/> constants.</param>
    /// <param name="detail">Optional free text, for an error worth logging on the device.</param>
    public async Task SendStatusAsync(string deviceId, string state, string? detail = null)
    {
        if (string.IsNullOrEmpty(deviceId) || !_sessions.TryGetValue(deviceId, out WakeSession? session))
        {
            return;
        }
        await SendStatusAsync(session, state, detail).ConfigureAwait(false);
    }

    /// <summary>Sends spoken audio to a satellite over the socket it already has open, paced so it arrives at
    /// about the rate the device plays it.
    ///
    /// <para>Speech reaches the device over HTTP today, which means a second connection and a second protocol
    /// per turn. This carries it on the wake socket instead, in the same header-plus-payload frames the device
    /// already sends audio in.</para>
    ///
    /// <para>Pacing is the part that is not obvious. The device has a small ring and paces an HTTP body by
    /// withholding TCP acknowledgements, but it cannot do that here without also stalling the <c>ping</c> and
    /// <c>detection</c> frames queued behind the audio on the same connection — and a satellite that stops
    /// answering pings is dropped after twenty seconds. So the server paces instead, writing a little faster
    /// than real time so the device's buffer fills rather than drains, and never faster than that.</para></summary>
    /// <param name="deviceId">The satellite to speak to. Unknown ids are ignored.</param>
    /// <param name="pcm">16-bit little-endian mono at <paramref name="sampleRate"/>.</param>
    /// <param name="sampleRate">Rate of the samples, for pacing.</param>
    /// <param name="cancel">Cancels mid-reply — which is what a barge-in looks like from here.</param>
    /// <returns>Bytes actually delivered.</returns>
    public async Task<int> SendAudioAsync(string deviceId, ReadOnlyMemory<byte> pcm, int sampleRate,
        CancellationToken cancel = default)
    {
        if (string.IsNullOrEmpty(deviceId) || !_sessions.TryGetValue(deviceId, out WakeSession? session))
        {
            return 0;
        }
        WakeFrameCodec? codec = session.Codec;
        if (codec is null || pcm.Length == 0)
        {
            return 0;
        }

        // 40 ms a frame: two of the device's own 20 ms audio frames, and comfortably under the 1280-byte
        // payloads its receive buffer is sized for.
        int frameBytes = Math.Max(2, sampleRate / 25 * 2);
        long sent = 0;
        long started = Environment.TickCount64;
        for (int offset = 0; offset < pcm.Length; offset += frameBytes)
        {
            cancel.ThrowIfCancellationRequested();
            int take = Math.Min(frameBytes, pcm.Length - offset);
            bool final = offset + take >= pcm.Length;
            string data = $"{{\"seq\":{offset / frameBytes},\"final\":{(final ? "true" : "false")}}}";
            try
            {
                await codec.WriteAsync("audio", data, pcm.Slice(offset, take), cancel).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                // The device hung up mid-reply. Everything before this point was already played.
                Logs.Verbose($"[Audio][Wake] Audio to '{deviceId}' stopped after {sent} bytes: {ex.Message}");
                return (int)sent;
            }
            sent += take;

            // Stay a little ahead of playback and no further. Sending at full speed would overrun the device's
            // ring; sending at exactly real time leaves it one late packet away from an underrun.
            double playedMs = sent / 2.0 / sampleRate * 1000.0;
            long elapsed = Environment.TickCount64 - started;
            int ahead = (int)(playedMs - elapsed) - AudioLeadMs;
            if (ahead > 0)
            {
                await Task.Delay(ahead, cancel).ConfigureAwait(false);
            }
        }
        return (int)sent;
    }

    /// <summary>How far ahead of real time the server is allowed to run when pushing audio, in milliseconds.
    /// Enough to absorb a late packet and a scheduling hiccup, small enough to fit any satellite ring worth
    /// having.</summary>
    private const int AudioLeadMs = 400;

    /// <summary>Sends a status frame on a session whose codec is already in hand.</summary>
    private static async Task SendStatusAsync(WakeSession session, string state, string? detail = null)
    {
        WakeFrameCodec? codec = session.Codec;
        if (codec is null) return;
        try
        {
            await codec.WriteAsync("status", WakeStatus.Data(state, detail), CancellationToken.None).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            // Advisory only. A device that dropped mid-turn will reconnect and get the next one.
            Logs.Verbose($"[Audio][Wake] Could not deliver status '{state}' to '{session.DeviceId}': {ex.Message}");
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
