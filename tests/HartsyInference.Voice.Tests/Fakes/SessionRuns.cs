using System.Text;
using HartsyInference.Engine.Services;

namespace HartsyInference.Voice.Tests.Fakes;

/// <summary>Drives caller audio through a real session and collects what it heard.</summary>
internal static class SessionRuns
{
    /// <summary>Pushes <paramref name="audio"/> and 1.5 s of silence into a session on <paramref name="models"/> answered by
    /// <paramref name="text"/>, waits for <paramref name="turns"/> turns and returns every user transcript, joined.</summary>
    public static async Task<(string Heard, List<VoiceTurnMetrics> Metrics)> HearAsync(VoiceModelSet models, ITextService text, float[] audio,
        int turns, Action<string> log, double secondsPerTurn = 120)
    {
        await using VoiceHarness harness = await VoiceHarness.StartAsync(models, text, TimeSpan.FromMilliseconds(2));
        harness.Push(audio);
        harness.PushSilence(1.5);
        StringBuilder heard = new();
        List<VoiceTurnMetrics> metrics = [];
        for (int turn = 1; turn <= turns; turn++)
        {
            VoiceAgentEvent done = await harness.TurnCompletedAsync(turn, secondsPerTurn);
            metrics.Add(done.Metrics!.Value);
            log(done.Metrics.Value.ToLogLine());
        }
        foreach (VoiceAgentEvent item in harness.Events.Where(e => e.Kind == VoiceAgentEventKind.UserTranscript))
        {
            log($"turn {item.TurnId} heard: \"{item.Text}\"");
            heard.Append(item.Text).Append(' ');
        }
        return (heard.ToString(), metrics);
    }
}
