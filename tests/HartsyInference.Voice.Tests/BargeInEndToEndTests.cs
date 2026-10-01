using HartsyInference.Engine;
using HartsyInference.Tests.Common;
using HartsyInference.Voice.Audio;
using HartsyInference.Voice.Tests.Fakes;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Voice.Tests;

/// <summary>Barge-in on the RTX 3060 with real models, paced like a call: caller audio arrives in 20 ms frames in real
/// time and the reply is read at 20 ms per frame. While Kokoro's reply plays, the caller starts speaking (JFK); the reply
/// must stop within 100 ms of the barge-in decision (the last reply audio read after it), the turn must be marked
/// interrupted, and the interrupting speech must be answered. In-process only: the gateway's flush crosses the socket in
/// the host's own test. Run alone, in a quiet window:
/// <c>CUDA_VISIBLE_DEVICES=1 HARTSY_REQUIRE_REAL_WEIGHTS=1 HARTSYINFERENCE_MODELS_DIR=/mnt/model-storage/Models dotnet test tests/HartsyInference.Voice.Tests -c Release --filter "FullyQualifiedName~BargeInEndToEndTests"</c>.</summary>
[Trait("Category", "GpuIntegration")]
[Trait("Category", "RealWeights")]
public sealed class BargeInEndToEndTests
{
    private const string LongReply = "Let me tell you about our opening hours. We open at nine in the morning on weekdays. "
        + "On Saturdays we open a little later, at ten. Sundays we are closed all day. Holidays follow the Sunday hours. "
        + "You can also reach us online at any time.";
    private const double StopGateMs = 100;

    private readonly ITestOutputHelper _output;

    public BargeInEndToEndTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task TheReplyStopsWithin100MsOfTheBargeIn()
    {
        if (!RealWeightGate.Require(_output.WriteLine, GpuVoiceRig.Assets()))
        {
            return;
        }
        using InferenceEngine? engine = GpuVoiceRig.OpenAudioEngine(_output.WriteLine);
        if (engine is null)
        {
            return;
        }
        VoiceAgentOptions options = GpuVoiceRig.Options(engine);
        await using VoiceModelSet models = await VoiceModelSet.LoadAsync(engine, options, VoiceAssets.WakeRoot);
        ScriptedTextService text = new ScriptedTextService { DefaultReply = "Okay, go ahead." }.Reply(LongReply);
        await models.WarmAsync(text);
        await using VoiceHarness harness = await VoiceHarness.StartAsync(models, text, TimeSpan.FromMilliseconds(20));
        float[] jfk = VoiceAssets.Jfk16k();
        using CancellationTokenSource stop = new();
        CallerAudio caller = new(harness.Session, jfk);
        Task pacing = caller.RunAsync(stop.Token);
        try
        {
            harness.Session.PushDtmf('1');
            await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.StateChanged && e.State == VoiceAgentState.Speaking, 60);
            await harness.Reader.WaitForSamplesAsync(16_000 * 3 / 2, 60);
            caller.StartSpeaking();

            VoiceAgentEvent bargeIn = await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.BargeIn, 30);
            VoiceTurnMetrics interrupted = (await harness.TurnCompletedAsync(1, 30)).Metrics!.Value;
            await harness.TurnCompletedAsync(2, 60);
            stop.Cancel();
            await pacing;

            long lastStaleNs = harness.Reader.Reads.Where(r => r.Ns >= bargeIn.TimestampNs && r.Ns <= bargeIn.TimestampNs + 1_000_000_000L)
                .Select(r => r.Ns).DefaultIfEmpty(bargeIn.TimestampNs).Max();
            double lastChunkMs = (lastStaleNs - bargeIn.TimestampNs) / 1e6;
            _output.WriteLine(interrupted.ToLogLine());
            _output.WriteLine($"barge-in on turn {bargeIn.TurnId}: last reply audio read {lastChunkMs:F1} ms after the decision; "
                + $"reader applied the flush {interrupted.BargeInStopMs:F1} ms after it");

            Assert.Equal(1, bargeIn.TurnId);
            Assert.True(interrupted.Interrupted);
            Assert.True(lastChunkMs <= StopGateMs, $"reply audio was still read {lastChunkMs:F1} ms after the barge-in.");
            Assert.True(interrupted.BargeInStopMs <= StopGateMs, $"the flush landed {interrupted.BargeInStopMs:F1} ms after the barge-in.");
            Assert.Contains(harness.Events, e => e.Kind == VoiceAgentEventKind.UserTranscript && e.TurnId == 2);
        }
        finally
        {
            stop.Cancel();
            await pacing;
        }
    }

    /// <summary>The caller's side of the line: 20 ms frames in real time, silence until told to speak, then the clip.</summary>
    private sealed class CallerAudio(VoiceAgentSession session, float[] speech)
    {
        private volatile bool _speaking;

        public void StartSpeaking() => _speaking = true;

        public async Task RunAsync(CancellationToken stop)
        {
            float[] silence = new float[VoiceAudioFrontend.FrameSamples];
            int offset = 0;
            using PeriodicTimer timer = new(TimeSpan.FromMilliseconds(20));
            try
            {
                while (await timer.WaitForNextTickAsync(stop))
                {
                    if (_speaking && offset + silence.Length <= speech.Length)
                    {
                        session.PushInbound(speech.AsSpan(offset, silence.Length));
                        offset += silence.Length;
                    }
                    else
                    {
                        session.PushInbound(silence);
                    }
                }
            }
            catch (OperationCanceledException) when (stop.IsCancellationRequested)
            {
                // The test is done with the caller.
            }
        }
    }
}
