using HartsyInference.Engine.Audio.Wake;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>Who owns the turn, stated on the transcript frame itself.
///
/// <para>The device cannot be told this any other way. A host that answers turns subscribes to
/// <c>Detected</c>, which is raised <em>after</em> the transcript frame has already gone out — so anything the
/// subscriber sends arrives at a satellite that has been running its own assistant call for several
/// milliseconds already. Both replies then land in one audio ring and play over each other. The flag has to
/// ride on the frame that starts the turn, and these tests pin the bytes because the parser on the other end is
/// in another repository and another language.</para></summary>
public sealed class WakeHandledFlagTests
{
    private static WakeEvent Event(string? transcript = "what time is it", string? command = "what time is it")
        => new() { DeviceId = "pico-1", Word = "hey_jarvis", Score = 0.9123f, Transcript = transcript, Command = command };

    [Fact]
    public void WithNoHostTurn_TheFrameIsUnchanged()
    {
        // A satellite talking to a server that answers nothing must see exactly what it saw before this
        // existed, byte for byte — otherwise every device in the field needs a reflash to keep working.
        Assert.Equal(
            "{\"name\":\"hey_jarvis\",\"score\":0.9123,\"transcript\":\"what time is it\",\"command\":\"what time is it\"}",
            WakeService.EventData(Event(), handled: false));
    }

    [Fact]
    public void WhenTheHostAnswers_TheFrameSaysSo()
    {
        Assert.Equal(
            "{\"name\":\"hey_jarvis\",\"score\":0.9123,\"transcript\":\"what time is it\",\"command\":\"what time is it\",\"handled\":true}",
            WakeService.EventData(Event(), handled: true));
    }

    [Fact]
    public void TheFlagIsLast_SoEveryOtherFieldKeepsItsPlace()
    {
        // Not cosmetic: a hand-written parser that scans for a key by name is indifferent to order, but one
        // that walks the string is not, and the satellite's json_helpers does a bit of both.
        string full = WakeService.EventData(
            new WakeEvent
            {
                DeviceId = "pico-1", Word = "hey_jarvis", Score = 1f, Route = "audiolab",
                Transcript = "t", Command = "c", Speaker = "kaleb",
            }, handled: true);
        Assert.Equal(
            "{\"name\":\"hey_jarvis\",\"score\":1.0000,\"route\":\"audiolab\",\"transcript\":\"t\",\"command\":\"c\","
            + "\"speaker\":\"kaleb\",\"handled\":true}", full);
    }

    [Fact]
    public void TheServiceDefaultsToAnsweringNothing()
    {
        // The device-side behaviour this unlocks is "do not run your own turn". Defaulting it on would silence
        // every existing satellite against a host that has no orchestrator.
        Assert.False(new WakeServiceOptions().HostHandlesTurns);
    }
}
