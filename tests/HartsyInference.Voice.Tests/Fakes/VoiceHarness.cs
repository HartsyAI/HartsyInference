using HartsyInference.Cpu;
using HartsyInference.Engine.Services;
using HartsyInference.Tools;
using HartsyInference.Voice.Audio;

namespace HartsyInference.Voice.Tests.Fakes;

/// <summary>A started <see cref="VoiceAgentSession"/> with an event recorder and helpers to feed caller audio and drain
/// the reply: on fake models (level-scripted VAD, <see cref="FakeSpeech"/>, <see cref="ScriptedTextService"/>, a CPU
/// "audio device"), or on a real <see cref="VoiceModelSet"/> the caller loaded.</summary>
internal sealed class VoiceHarness : IAsyncDisposable
{
    public const float SpeechLevel = 0.9f;
    public const int Rate = VoiceAudioFrontend.SampleRate;

    private readonly List<VoiceAgentEvent> _events = [];
    private readonly SemaphoreSlim _eventArrived = new(0);
    private readonly CpuBackend? _device;
    private readonly bool _ownsModels;
    private readonly FakeSpeech? _speech;
    private OutboundReader? _reader;

    private VoiceHarness(VoiceModelSet models, bool ownsModels, VoiceAgentOptions options, FakeSpeech? speech, ITextService text,
        ToolRegistry tools, int outboundCapacity, CpuBackend? device)
    {
        Options = options;
        _speech = speech;
        TextService = text;
        Tools = tools;
        Models = models;
        _ownsModels = ownsModels;
        _device = device;
        Session = new VoiceAgentSession(Models, text, tools, options, inboundCapacity: 0, outboundCapacity);
        Session.EventRaised += Record;
    }

    public VoiceAgentOptions Options { get; }

    public FakeSpeech Speech => _speech ?? throw new InvalidOperationException("This harness runs real speech models.");

    public ITextService TextService { get; }

    public ScriptedTextService Text => TextService as ScriptedTextService ?? throw new InvalidOperationException("This harness runs a real language model.");

    public ToolRegistry Tools { get; }

    public VoiceModelSet Models { get; }

    public VoiceAgentSession Session { get; }

    public OutboundReader Reader => _reader ?? throw new InvalidOperationException("No reader started.");

    /// <summary>Defaults for the harness: synthesis and playback at the same rate, so played samples are the markers.</summary>
    public static VoiceAgentOptions DefaultOptions() => new() { OutboundSampleRate = 24_000, LlmDevice = "cpu", AudioDevice = "cpu" };

    /// <summary>A session on fake models.</summary>
    public static async Task<VoiceHarness> StartAsync(VoiceAgentOptions? options = null, FakeSpeech? speech = null, ScriptedTextService? text = null,
        ToolRegistry? tools = null, int outboundCapacity = 0, bool startReader = true, TimeSpan? readerPause = null)
    {
        VoiceAgentOptions resolved = options ?? DefaultOptions();
        FakeSpeech fake = speech ?? new FakeSpeech();
        CpuBackend device = new();
        VoiceModelSet models = new(resolved, fake, device, () => new LevelVadModel(), createDenoiser: null);
        VoiceHarness harness = new(models, ownsModels: true, resolved, fake, text ?? new ScriptedTextService(), tools ?? new ToolRegistry(),
            outboundCapacity, device);
        return await harness.BeginAsync(startReader, readerPause ?? TimeSpan.FromMilliseconds(1));
    }

    /// <summary>A session on real models; the caller keeps ownership of <paramref name="models"/>.</summary>
    public static async Task<VoiceHarness> StartAsync(VoiceModelSet models, ITextService text, TimeSpan readerPause, ToolRegistry? tools = null,
        VoiceAgentOptions? options = null)
    {
        VoiceHarness harness = new(models, ownsModels: false, options ?? models.Options, speech: null, text, tools ?? new ToolRegistry(),
            outboundCapacity: 0, device: null);
        return await harness.BeginAsync(startReader: true, readerPause);
    }

    public OutboundReader StartReader(TimeSpan pause, int chunk = 320)
    {
        _reader = new OutboundReader(Session, chunk, pause);
        return _reader;
    }

    public IReadOnlyList<VoiceAgentEvent> Events
    {
        get
        {
            lock (_events)
            {
                return [.. _events];
            }
        }
    }

    /// <summary>Pushes <paramref name="seconds"/> of audio at a constant level, in 20 ms frames.</summary>
    public void Push(double seconds, float level)
    {
        int total = (int)Math.Round(seconds * Rate);
        float[] frame = new float[VoiceAudioFrontend.FrameSamples];
        Array.Fill(frame, level);
        for (int sent = 0; sent < total; sent += frame.Length)
        {
            Session.PushInbound(frame.AsSpan(0, Math.Min(frame.Length, total - sent)));
        }
    }

    /// <summary>Pushes <paramref name="audio"/> in 20 ms frames, as fast as it goes.</summary>
    public void Push(ReadOnlySpan<float> audio)
    {
        for (int sent = 0; sent < audio.Length; sent += VoiceAudioFrontend.FrameSamples)
        {
            Session.PushInbound(audio.Slice(sent, Math.Min(VoiceAudioFrontend.FrameSamples, audio.Length - sent)));
        }
    }

    public void PushSpeech(double seconds) => Push(seconds, SpeechLevel);

    public void PushSilence(double seconds) => Push(seconds, 0f);

    /// <summary>Waits for the first recorded event matching <paramref name="match"/>.</summary>
    public async Task<VoiceAgentEvent> WaitForAsync(Func<VoiceAgentEvent, bool> match, double seconds = 20)
    {
        DateTime deadline = DateTime.UtcNow.AddSeconds(seconds);
        while (true)
        {
            lock (_events)
            {
                foreach (VoiceAgentEvent item in _events)
                {
                    if (match(item))
                    {
                        return item;
                    }
                }
            }
            TimeSpan left = deadline - DateTime.UtcNow;
            if (left <= TimeSpan.Zero || !await _eventArrived.WaitAsync(left))
            {
                throw new TimeoutException($"No matching event within {seconds} s; saw: {string.Join(", ", Events.Select(e => e.Kind + (e.Kind == VoiceAgentEventKind.StateChanged ? ":" + e.State : "")))}");
            }
        }
    }

    public Task<VoiceAgentEvent> TurnCompletedAsync(int turnId, double seconds = 20) =>
        WaitForAsync(e => e.Kind == VoiceAgentEventKind.TurnCompleted && e.TurnId == turnId, seconds);

    public async ValueTask DisposeAsync()
    {
        await Session.EndAsync().WaitAsync(TimeSpan.FromSeconds(60));
        if (_reader is not null)
        {
            await _reader.DisposeAsync();
        }
        if (_ownsModels)
        {
            await Models.DisposeAsync();
        }
        _device?.Dispose();
    }

    private async Task<VoiceHarness> BeginAsync(bool startReader, TimeSpan readerPause)
    {
        await Session.StartAsync().WaitAsync(TimeSpan.FromSeconds(30));
        if (startReader)
        {
            StartReader(readerPause);
        }
        return this;
    }

    private void Record(VoiceAgentEvent item)
    {
        lock (_events)
        {
            _events.Add(item);
        }
        _eventArrived.Release();
    }
}
