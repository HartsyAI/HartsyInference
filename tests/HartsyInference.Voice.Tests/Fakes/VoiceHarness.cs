using HartsyInference.Audio.Models.Wake;
using HartsyInference.Cpu;
using HartsyInference.Tools;
using HartsyInference.Voice.Audio;

namespace HartsyInference.Voice.Tests.Fakes;

/// <summary>A started <see cref="VoiceAgentSession"/> on fake models (level-scripted VAD, <see cref="FakeSpeech"/>,
/// <see cref="ScriptedTextService"/>) with a CPU "audio device", an event recorder and helpers to feed caller audio and
/// drain the reply.</summary>
internal sealed class VoiceHarness : IAsyncDisposable
{
    public const float SpeechLevel = 0.9f;
    public const int Rate = VoiceAudioFrontend.SampleRate;

    private readonly List<VoiceAgentEvent> _events = [];
    private readonly SemaphoreSlim _eventArrived = new(0);
    private readonly CpuBackend _device = new();
    private OutboundReader? _reader;

    private VoiceHarness(VoiceAgentOptions options, FakeSpeech speech, ScriptedTextService text, ToolRegistry tools, int outboundCapacity)
    {
        Options = options;
        Speech = speech;
        Text = text;
        Tools = tools;
        Models = new VoiceModelSet(options, speech, _device, () => new LevelVadModel(), createDenoiser: null);
        Session = new VoiceAgentSession(Models, text, tools, options, inboundCapacity: 0, outboundCapacity);
        Session.EventRaised += Record;
    }

    public VoiceAgentOptions Options { get; }

    public FakeSpeech Speech { get; }

    public ScriptedTextService Text { get; }

    public ToolRegistry Tools { get; }

    public VoiceModelSet Models { get; }

    public VoiceAgentSession Session { get; }

    public OutboundReader Reader => _reader ?? throw new InvalidOperationException("No reader started.");

    /// <summary>Defaults for the harness: synthesis and playback at the same rate, so played samples are the markers.</summary>
    public static VoiceAgentOptions DefaultOptions() => new() { OutboundSampleRate = 24_000, LlmDevice = "cpu", AudioDevice = "cpu" };

    public static async Task<VoiceHarness> StartAsync(VoiceAgentOptions? options = null, FakeSpeech? speech = null, ScriptedTextService? text = null,
        ToolRegistry? tools = null, int outboundCapacity = 0, bool startReader = true, TimeSpan? readerPause = null)
    {
        VoiceHarness harness = new(options ?? DefaultOptions(), speech ?? new FakeSpeech(), text ?? new ScriptedTextService(), tools ?? new ToolRegistry(), outboundCapacity);
        await harness.Session.StartAsync().WaitAsync(TimeSpan.FromSeconds(30));
        if (startReader)
        {
            harness.StartReader(readerPause ?? TimeSpan.FromMilliseconds(1));
        }
        return harness;
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
        await Session.EndAsync().WaitAsync(TimeSpan.FromSeconds(30));
        if (_reader is not null)
        {
            await _reader.DisposeAsync();
        }
        await Models.DisposeAsync();
        _device.Dispose();
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
