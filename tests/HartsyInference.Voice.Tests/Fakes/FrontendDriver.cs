using HartsyInference.Cpu;
using HartsyInference.Voice.Audio;
using HartsyInference.Voice.Turns;

namespace HartsyInference.Voice.Tests.Fakes;

/// <summary>A <see cref="VoiceAudioFrontend"/> on the level-scripted VAD, driven frame by frame on the test thread, with
/// every non-empty result recorded against the front-end's sample clock.</summary>
internal sealed class FrontendDriver : IDisposable
{
    public const int Rate = VoiceAudioFrontend.SampleRate;
    public const int Frame = VoiceAudioFrontend.FrameSamples;

    private readonly CpuBackend _cpu = new();
    private readonly float[] _frame = new float[Frame];

    public FrontendDriver(VoiceAgentOptions? options = null, VoiceTurnSignals? signals = null)
    {
        Options = options ?? new VoiceAgentOptions();
        Signals = signals ?? new VoiceTurnSignals();
        Frontend = new VoiceAudioFrontend(_cpu, new LevelVadModel(), null, Signals, Options);
    }

    public VoiceAgentOptions Options { get; }

    public VoiceTurnSignals Signals { get; }

    public VoiceAudioFrontend Frontend { get; }

    /// <summary>Samples fed so far (the input clock; equals the front-end clock minus the partial VAD window).</summary>
    public long Fed { get; private set; }

    public List<Decision> Decisions { get; } = [];

    /// <summary>Feeds <paramref name="seconds"/> of constant <paramref name="level"/>.</summary>
    public void Feed(double seconds, float level) => Feed((int)Math.Round(seconds * Rate), _ => level);

    /// <summary>Feeds <paramref name="samples"/> samples whose values come from <paramref name="sample"/>(input index).</summary>
    public void Feed(int samples, Func<long, float> sample)
    {
        if (samples % Frame != 0)
        {
            throw new ArgumentException($"Feed whole {Frame}-sample frames.", nameof(samples));
        }
        for (int done = 0; done < samples; done += Frame)
        {
            for (int i = 0; i < Frame; i++)
            {
                _frame[i] = sample(Fed + i);
            }
            Fed += Frame;
            VoiceFrameEvents events = Frontend.ProcessFrame(_frame);
            if (events != VoiceFrameEvents.None)
            {
                Decisions.Add(new Decision(events, Frontend.ClockSamples, Frontend.UtteranceStartSample, Frontend.UtteranceSamples,
                    Frontend.HangoverSamples, Frontend.BargeInTurn));
            }
        }
    }

    public IEnumerable<Decision> With(VoiceFrameEvents kind) => Decisions.Where(d => (d.Events & kind) != 0);

    public void Dispose()
    {
        Frontend.Dispose();
        _cpu.Dispose();
    }

    public readonly record struct Decision(VoiceFrameEvents Events, long Clock, long UtteranceStart, int UtteranceSamples, long HangoverSamples,
        int BargeInTurn);
}
