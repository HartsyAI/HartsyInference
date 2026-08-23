using System.Buffers.Binary;
using System.Text.Json;
using HartsyInference.Audio.Io;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Audio.Wake.Wyoming;

/// <summary>Serves one Home Assistant connection.
///
/// <para>Unlike the satellite listener there is no handshake and no session identity: Home Assistant opens a fresh
/// connection per operation, sends <c>describe</c> or a domain request immediately, reads the reply and drops the
/// socket. Every field of state here therefore belongs to a single operation.</para></summary>
internal sealed class WyomingConnection : IDisposable
{
    private readonly IInferenceEngine? _engine;
    private readonly WyomingOptions _options;
    private readonly WyomingFrameCodec _codec;
    private readonly string _remote;
    private readonly MemoryStream _utterance = new();

    private StreamMode _mode = StreamMode.Idle;
    private IWyomingWakeDetector? _detector;
    private WyomingArtifact? _asrModel;
    private string? _language;
    private bool _detectionSent;
    private bool _utteranceTruncated;
    private long _scoredSamples;
    private int _rate = 16_000;
    private int _width = 2;
    private int _channels = 1;

    public WyomingConnection(IInferenceEngine? engine, WyomingOptions options, WyomingFrameCodec codec, string remote)
    {
        _engine = engine;
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _codec = codec ?? throw new ArgumentNullException(nameof(codec));
        _remote = remote;
    }

    /// <summary>Reads and answers events until the peer closes or the host stops.</summary>
    public async Task RunAsync(CancellationToken cancel)
    {
        while (!cancel.IsCancellationRequested)
        {
            WyomingEvent? evt = await _codec.ReadAsync(cancel).ConfigureAwait(false);
            if (evt is null) return;
            using (evt)
            {
                await HandleAsync(evt, cancel).ConfigureAwait(false);
            }
        }
    }

    private async Task HandleAsync(WyomingEvent evt, CancellationToken cancel)
    {
        switch (evt.Type)
        {
            case "describe":
                await _codec.WriteAsync("info", WyomingInfo.Build(_options, _options.WakeDetectorFactory is not null), cancel).ConfigureAwait(false);
                break;
            case "ping":
                await _codec.WriteAsync("pong", Echo(evt.GetString("text")), cancel).ConfigureAwait(false);
                break;
            case "transcribe":
                StartTranscribe(evt);
                break;
            case "detect":
                StartDetect(evt);
                break;
            case "synthesize":
                await SynthesizeAsync(evt, cancel).ConfigureAwait(false);
                break;
            case "audio-start":
                StartAudio(evt);
                break;
            case "audio-chunk":
                await ConsumeAudioAsync(evt, cancel).ConfigureAwait(false);
                break;
            case "audio-stop":
                await FinishAudioAsync(cancel).ConfigureAwait(false);
                break;
            case "select-program":
                // One program per domain is advertised, so the selection can only ever resolve to it.
                Logs.Verbose($"[Audio][Wyoming] {_remote} selected program '{evt.GetString("name")}'.");
                break;
            default:
                Logs.Verbose($"[Audio][Wyoming] Ignoring unsupported event '{evt.Type}' from {_remote}.");
                break;
        }
    }

    #region Speech-to-text

    private void StartTranscribe(WyomingEvent evt)
    {
        _mode = StreamMode.Transcribe;
        _language = evt.GetString("language");
        string? name = evt.GetString("name");
        _asrModel = Resolve(_options.AsrModels, name);
        ResetUtterance();
        if (name is not null && _asrModel is not null && !string.Equals(name, _asrModel.Name, StringComparison.OrdinalIgnoreCase))
            Logs.Warning($"[Audio][Wyoming] {_remote} asked for ASR model '{name}', which is not advertised; using '{_asrModel.Name}'.");
    }

    private void StartAudio(WyomingEvent evt)
    {
        _rate = evt.GetInt32("rate") ?? _rate;
        _width = evt.GetInt32("width") ?? _width;
        _channels = evt.GetInt32("channels") ?? _channels;
        ResetUtterance();
        _detector?.Reset();
        _detectionSent = false;
        _scoredSamples = 0;
    }

    private async Task ConsumeAudioAsync(WyomingEvent evt, CancellationToken cancel)
    {
        // Format lives on every chunk as well as on audio-start, and Home Assistant does not always send the
        // latter — take it from whichever arrives.
        _rate = evt.GetInt32("rate") ?? _rate;
        _width = evt.GetInt32("width") ?? _width;
        _channels = evt.GetInt32("channels") ?? _channels;
        if (evt.Payload is null || evt.PayloadLength == 0) return;

        switch (_mode)
        {
            case StreamMode.Transcribe:
                if (_utterance.Length + evt.PayloadLength > _options.MaxUtteranceBytes)
                {
                    if (!_utteranceTruncated)
                        Logs.Warning($"[Audio][Wyoming] {_remote} exceeded {_options.MaxUtteranceBytes} buffered bytes without an audio-stop; dropping the rest of the utterance.");
                    _utteranceTruncated = true;
                    return;
                }
                _utterance.Write(evt.Payload, 0, evt.PayloadLength);
                break;
            case StreamMode.Detect:
                await ScoreAsync(evt, cancel).ConfigureAwait(false);
                break;
        }
    }

    private async Task FinishAudioAsync(CancellationToken cancel)
    {
        switch (_mode)
        {
            case StreamMode.Transcribe:
                await TranscribeAsync(cancel).ConfigureAwait(false);
                break;
            case StreamMode.Detect:
                if (!_detectionSent)
                    await _codec.WriteAsync("not-detected", ReadOnlyMemory<byte>.Empty, cancel).ConfigureAwait(false);
                break;
        }
        _mode = StreamMode.Idle;
    }

    private async Task TranscribeAsync(CancellationToken cancel)
    {
        if (_engine is null)
        {
            await FailAsync("No inference engine is attached to this endpoint.", "no-engine", cancel).ConfigureAwait(false);
            return;
        }
        if (_asrModel is null)
        {
            await FailAsync("No speech-to-text model is configured.", "no-model", cancel).ConfigureAwait(false);
            return;
        }
        if (_width != 2 || _channels != 1)
        {
            await FailAsync($"Expected 16-bit mono audio, got {_width * 8}-bit with {_channels} channel(s).", "bad-format", cancel).ConfigureAwait(false);
            return;
        }
        if (_utterance.Length < 2)
        {
            await _codec.WriteAsync("transcript", Transcript("", _language ?? _options.DefaultLanguage), cancel).ConfigureAwait(false);
            return;
        }

        try
        {
            // The engine decodes whatever rate the WAV declares, so the incoming rate is carried through rather
            // than resampled here.
            byte[] wav = ToWav(_utterance.GetBuffer().AsSpan(0, (int)_utterance.Length), _rate);
            ModelSpec spec = Registry.ModelResolver.Resolve(_asrModel.ResolvedModelId, null, Modality.Transcribe);
            AudioRequest request = new()
            {
                Audio = new AudioClip { Data = wav, Format = "wav" },
                Language = _language ?? _options.DefaultLanguage,
            };
            Task<TranscriptResult> Run() => _engine.Transcribe.RunAsync(spec, request, cancel);
            TranscriptResult result = _options.TranscribeGate is null
                ? await Run().ConfigureAwait(false)
                : await _options.TranscribeGate(Run).ConfigureAwait(false);
            await _codec.WriteAsync("transcript", Transcript(result.Text, result.Language), cancel).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio][Wyoming] Transcription for {_remote} failed.", ex);
            await FailAsync(ex.Message, "transcribe-failed", cancel).ConfigureAwait(false);
        }
        finally
        {
            ResetUtterance();
        }
    }

    #endregion

    #region Wake

    private void StartDetect(WyomingEvent evt)
    {
        _mode = StreamMode.Detect;
        _detectionSent = false;
        _scoredSamples = 0;
        IReadOnlyList<string> names = evt.GetStringArray("names") ?? [];
        if (_options.WakeDetectorFactory is null)
        {
            Logs.Warning($"[Audio][Wyoming] {_remote} asked to detect [{string.Join(", ", names)}] but no wake detector is wired; every stream will answer not-detected.");
            return;
        }
        try
        {
            _detector?.Dispose();
            _detector = _options.WakeDetectorFactory(names);
        }
        catch (Exception ex)
        {
            _detector = null;
            Logs.Error($"[Audio][Wyoming] Could not build a wake detector for {_remote}.", ex);
        }
    }

    private async Task ScoreAsync(WyomingEvent evt, CancellationToken cancel)
    {
        if (_detector is null || _detectionSent || _width != 2 || _channels != 1) return;
        int samples = evt.PayloadLength / 2;
        float[] pcm = new float[samples];
        ReadOnlySpan<byte> bytes = evt.Payload!.AsSpan(0, samples * 2);
        // Wake models score int16-scaled audio; normalizing to ±1 here would silently mis-score every stream.
        for (int i = 0; i < samples; i++) pcm[i] = BinaryPrimitives.ReadInt16LittleEndian(bytes[(i * 2)..]);
        _scoredSamples += samples;

        WyomingWakeHit? hit;
        try
        {
            hit = _detector.Push(pcm);
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio][Wyoming] Wake scoring for {_remote} failed; the stream continues unscored.", ex);
            _detector.Dispose();
            _detector = null;
            return;
        }
        if (hit is not WyomingWakeHit fired) return;

        _detectionSent = true;
        long timestamp = (long)(_scoredSamples / (double)Math.Max(1, _rate) * 1000);
        await _codec.WriteAsync("detection", WyomingFrameCodec.BuildData(writer =>
        {
            writer.WriteString("name", fired.Name);
            writer.WriteNumber("timestamp", timestamp);
        }), cancel).ConfigureAwait(false);
        Logs.Info($"[Audio][Wyoming] Wake word '{fired.Name}' fired for {_remote} at {fired.Score:F4}.");
    }

    #endregion

    #region Text-to-speech

    private async Task SynthesizeAsync(WyomingEvent evt, CancellationToken cancel)
    {
        string text = evt.GetString("text") ?? "";
        JsonElement? voice = evt.GetObject("voice");
        WyomingArtifact? artifact = Resolve(_options.TtsVoices, StringOf(voice, "name"));

        if (_engine is null)
        {
            await FailAsync("No inference engine is attached to this endpoint.", "no-engine", cancel).ConfigureAwait(false);
            return;
        }
        if (artifact is null)
        {
            await FailAsync("No text-to-speech voice is configured.", "no-voice", cancel).ConfigureAwait(false);
            return;
        }
        if (text.Length == 0)
        {
            await FailAsync("synthesize carried no text.", "no-text", cancel).ConfigureAwait(false);
            return;
        }

        try
        {
            ModelSpec spec = Registry.ModelResolver.Resolve(artifact.ResolvedModelId, null, Modality.Speech);
            SpeechRequest request = new() { Text = text, Voice = artifact.VoiceName ?? StringOf(voice, "speaker") };
            Task<AudioResult> Run() => _engine.Speech.SynthesizeAsync(spec, request, cancel);
            AudioResult result = _options.SynthesizeGate is null
                ? await Run().ConfigureAwait(false)
                : await _options.SynthesizeGate(Run).ConfigureAwait(false);
            await SendAudioAsync(result, cancel).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            Logs.Error($"[Audio][Wyoming] Synthesis for {_remote} failed.", ex);
            await FailAsync(ex.Message, "synthesize-failed", cancel).ConfigureAwait(false);
        }
    }

    /// <summary>Streams a rendered clip out as Wyoming audio at its own sample rate — Home Assistant resamples, so forcing 16 kHz here would only add a lossy step the wire format does not require.</summary>
    private async Task SendAudioAsync(AudioResult result, CancellationToken cancel)
    {
        if (!string.Equals(result.Format, "wav", StringComparison.OrdinalIgnoreCase))
            throw new HartsyInferenceException($"Wyoming needs raw PCM, and only the engine's WAV output can be unwrapped to it; the pipeline returned '{result.Format}'.");

        using MemoryStream source = new(result.Data);
        WavFile.DecodedAudio decoded = WavFile.Read(source);
        float[] mono = decoded.ToMono();
        int rate = decoded.SampleRate > 0 ? decoded.SampleRate : result.SampleRate;

        byte[] pcm = new byte[mono.Length * 2];
        for (int i = 0; i < mono.Length; i++)
        {
            float clamped = Math.Clamp(mono[i], -1f, 1f);
            BinaryPrimitives.WriteInt16LittleEndian(pcm.AsSpan(i * 2), (short)Math.Round(clamped * 32767f));
        }

        await _codec.WriteAsync("audio-start", Format(rate, 0), cancel).ConfigureAwait(false);
        int chunk = Math.Max(2, _options.SynthesisChunkBytes & ~1);
        for (int offset = 0; offset < pcm.Length; offset += chunk)
        {
            int length = Math.Min(chunk, pcm.Length - offset);
            long timestamp = (long)(offset / 2.0 / rate * 1000);
            await _codec.WriteAsync("audio-chunk", Format(rate, timestamp), pcm.AsMemory(offset, length), cancel).ConfigureAwait(false);
        }
        await _codec.WriteAsync("audio-stop", WyomingFrameCodec.BuildData(writer =>
            writer.WriteNumber("timestamp", (long)(mono.Length / (double)rate * 1000))), cancel).ConfigureAwait(false);
    }

    private static string? StringOf(JsonElement? container, string key)
    {
        if (container is not JsonElement element || element.ValueKind != JsonValueKind.Object) return null;
        return element.TryGetProperty(key, out JsonElement value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
    }

    private static byte[] Format(int rate, long timestamp) => WyomingFrameCodec.BuildData(writer =>
    {
        writer.WriteNumber("rate", rate);
        writer.WriteNumber("width", 2);
        writer.WriteNumber("channels", 1);
        writer.WriteNumber("timestamp", timestamp);
    });

    #endregion

    private static WyomingArtifact? Resolve(IReadOnlyList<WyomingArtifact> artifacts, string? name)
    {
        if (artifacts.Count == 0) return null;
        if (name is null) return artifacts[0];
        foreach (WyomingArtifact artifact in artifacts)
        {
            if (string.Equals(artifact.Name, name, StringComparison.OrdinalIgnoreCase)) return artifact;
        }
        return artifacts[0];
    }

    private static byte[] Echo(string? text) =>
        text is null ? [] : WyomingFrameCodec.BuildData(writer => writer.WriteString("text", text));

    private static byte[] Transcript(string text, string language) => WyomingFrameCodec.BuildData(writer =>
    {
        writer.WriteString("text", text);
        writer.WriteString("language", language);
    });

    private static byte[] ToWav(ReadOnlySpan<byte> pcm, int rate)
    {
        int samples = pcm.Length / 2;
        float[] normalized = new float[samples];
        for (int i = 0; i < samples; i++) normalized[i] = BinaryPrimitives.ReadInt16LittleEndian(pcm[(i * 2)..]) / 32768f;
        using MemoryStream ms = new();
        WavFile.WriteMono16(ms, normalized, rate);
        return ms.ToArray();
    }

    private async Task FailAsync(string text, string code, CancellationToken cancel)
    {
        try
        {
            await _codec.WriteAsync("error", WyomingFrameCodec.BuildData(writer =>
            {
                writer.WriteString("text", text);
                writer.WriteString("code", code);
            }), cancel).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            // The peer usually drops the socket the moment a request fails, so this is expected noise.
            Logs.Verbose($"[Audio][Wyoming] Could not report '{code}' to {_remote}: {ex.Message}");
        }
    }

    private void ResetUtterance()
    {
        _utterance.SetLength(0);
        _utteranceTruncated = false;
    }

    public void Dispose()
    {
        _detector?.Dispose();
        _utterance.Dispose();
    }

    private enum StreamMode
    {
        Idle,
        Transcribe,
        Detect,
    }
}
