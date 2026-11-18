using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.ToolCalling;
using HartsyInference.Voice.Tests.Fakes;
using Xunit;

namespace HartsyInference.Voice.Tests;

/// <summary>A message the caller typed is a turn like a spoken one: same events and history, no recognition timings, it
/// interrupts an audible reply when barge-in is on and queues behind it when it is off, and it is refused where nothing
/// could answer it. A transcribe-only session reports what it hears and calls neither the model nor the synthesizer.</summary>
public sealed class VoiceTypedTurnTests
{
    private static VoiceAgentOptions Options(bool bargeIn = true, bool transcribeOnly = false) =>
        VoiceHarness.DefaultOptions() with { BargeInEnabled = bargeIn, TranscribeOnly = transcribeOnly };

    [Fact]
    public async Task ATypedMessageIsAnsweredAndSpokenLikeAnUtterance()
    {
        await using VoiceHarness harness = await VoiceHarness.StartAsync(Options(), text: new ScriptedTextService().Reply("It is noon."));

        await harness.Session.SubmitUserTextAsync("  what time is it  ").WaitAsync(TimeSpan.FromSeconds(20));

        VoiceAgentEvent user = await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.UserTranscript);
        VoiceAgentEvent assistant = await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.AssistantTranscript);
        VoiceAgentEvent done = await harness.TurnCompletedAsync(user.TurnId);
        Assert.Equal("what time is it", user.Text);
        Assert.Equal("It is noon.", assistant.Text);
        Assert.Equal(user.TurnId, assistant.TurnId);
        Assert.Equal(VoiceTurnKind.Typed, done.Metrics!.Value.Kind);
        Assert.Null(done.Metrics.Value.SttMs);
        Assert.Null(done.Metrics.Value.EndpointMs);
        Assert.NotNull(done.Metrics.Value.TotalMs);
        TextRequest request = Assert.Single(harness.Text.Requests);
        Assert.Contains(request.Messages, m => m.Role == TextRole.User && m.Content == "what time is it");
        Assert.Empty(harness.Speech.TranscribedSamples);
        Assert.NotEmpty(harness.Speech.Synthesized);
        Assert.Contains(harness.Session.Transcript, t => t.Role == TextRole.User && t.Text == "what time is it");
    }

    [Fact]
    public async Task TwoQuickMessagesRunInOrder()
    {
        ScriptedTextService text = new ScriptedTextService().Reply("First answer.").Reply("Second answer.");
        await using VoiceHarness harness = await VoiceHarness.StartAsync(Options(bargeIn: false), text: text);

        Task first = harness.Session.SubmitUserTextAsync("one");
        Task second = harness.Session.SubmitUserTextAsync("two");
        await Task.WhenAll(first, second).WaitAsync(TimeSpan.FromSeconds(30));

        string[] heard = [.. harness.Events.Where(e => e.Kind == VoiceAgentEventKind.UserTranscript).Select(e => e.Text!)];
        string[] answers = [.. harness.Events.Where(e => e.Kind == VoiceAgentEventKind.AssistantTranscript).Select(e => e.Text!)];
        Assert.Equal(["one", "two"], heard);
        Assert.Equal(["First answer.", "Second answer."], answers);
    }

    [Fact]
    public async Task ATypedMessageInterruptsAnAudibleReplyWhenBargeInIsOn()
    {
        ScriptedTextService text = new ScriptedTextService().Reply("A long first answer that keeps going.").Reply("The second answer.");
        await using VoiceHarness harness = await VoiceHarness.StartAsync(Options(bargeIn: true), text: text);
        harness.Reader.Paused = true;

        Task first = harness.Session.SubmitUserTextAsync("first");
        await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.StateChanged && e.State == VoiceAgentState.Speaking);
        Task second = harness.Session.SubmitUserTextAsync("second");
        VoiceAgentEvent bargeIn = await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.BargeIn);
        harness.Reader.Paused = false;
        await second.WaitAsync(TimeSpan.FromSeconds(30));
        await first.WaitAsync(TimeSpan.FromSeconds(30));

        VoiceAgentEvent firstDone = await harness.TurnCompletedAsync(bargeIn.TurnId);
        Assert.True(firstDone.Metrics!.Value.Interrupted);
        VoiceAgentEvent secondUser = await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.UserTranscript && e.Text == "second");
        Assert.True(secondUser.TurnId > bargeIn.TurnId);
        VoiceAgentEvent secondDone = await harness.TurnCompletedAsync(secondUser.TurnId);
        Assert.False(secondDone.Metrics!.Value.Interrupted);
    }

    [Fact]
    public async Task ATypedMessageWaitsForAnAudibleReplyWhenBargeInIsOff()
    {
        ScriptedTextService text = new ScriptedTextService().Reply("A long first answer that keeps going.").Reply("The second answer.");
        await using VoiceHarness harness = await VoiceHarness.StartAsync(Options(bargeIn: false), text: text);
        harness.Reader.Paused = true;

        Task first = harness.Session.SubmitUserTextAsync("first");
        await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.StateChanged && e.State == VoiceAgentState.Speaking);
        Task second = harness.Session.SubmitUserTextAsync("second");
        await Task.Delay(300);

        Assert.DoesNotContain(harness.Events, e => e.Kind == VoiceAgentEventKind.BargeIn);
        Assert.DoesNotContain(harness.Events, e => e.Kind == VoiceAgentEventKind.UserTranscript && e.Text == "second");
        harness.Reader.Paused = false;
        await Task.WhenAll(first, second).WaitAsync(TimeSpan.FromSeconds(30));
        VoiceAgentEvent firstUser = await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.UserTranscript && e.Text == "first");
        Assert.False((await harness.TurnCompletedAsync(firstUser.TurnId)).Metrics!.Value.Interrupted);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("?!...")]
    public async Task AMessageWithNoWordsIsRefused(string message)
    {
        await using VoiceHarness harness = await VoiceHarness.StartAsync(Options());

        await Assert.ThrowsAsync<ArgumentException>(() => harness.Session.SubmitUserTextAsync(message));
        Assert.Empty(harness.Text.Requests);
    }

    [Fact]
    public async Task AMessageOverTheLimitIsRefusedAndOneAtTheLimitIsNot()
    {
        await using VoiceHarness harness = await VoiceHarness.StartAsync(Options(), text: new ScriptedTextService { DefaultReply = "Ok." });

        await Assert.ThrowsAsync<ArgumentException>(() => harness.Session.SubmitUserTextAsync(new string('a', VoiceAgentSession.MaxTypedTextChars + 1)));
        await harness.Session.SubmitUserTextAsync(new string('a', VoiceAgentSession.MaxTypedTextChars)).WaitAsync(TimeSpan.FromSeconds(30));
    }

    [Fact]
    public async Task AMessageBeforeStartIsRefused()
    {
        VoiceAgentOptions options = Options();
        using CpuBackend device = new();
        await using VoiceModelSet models = new(options, new FakeSpeech(), device, () => new LevelVadModel(), createDenoiser: null);
        await using VoiceAgentSession session = new(models, new ScriptedTextService(), new ToolRegistry(), options);

        await Assert.ThrowsAsync<InvalidOperationException>(() => session.SubmitUserTextAsync("hello"));
    }

    [Fact]
    public async Task AMessageAfterTheSessionEndedIsRefused()
    {
        await using VoiceHarness harness = await VoiceHarness.StartAsync(Options());
        await harness.Session.EndAsync().WaitAsync(TimeSpan.FromSeconds(30));

        await Assert.ThrowsAsync<InvalidOperationException>(() => harness.Session.SubmitUserTextAsync("hello"));
    }

    [Fact]
    public async Task ATranscribeOnlySessionReportsWhatItHearsWithoutAskingTheModelOrSpeaking()
    {
        FakeSpeech speech = new();
        speech.Transcripts.Enqueue("open the mixer");
        await using VoiceHarness harness = await VoiceHarness.StartAsync(Options(transcribeOnly: true), speech);

        harness.PushSpeech(0.8);
        harness.PushSilence(1.2);

        VoiceAgentEvent heard = await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.UserTranscript);
        await harness.TurnCompletedAsync(heard.TurnId);
        Assert.Equal("open the mixer", heard.Text);
        Assert.DoesNotContain(harness.Events, e => e.Kind == VoiceAgentEventKind.AssistantTranscript);
        Assert.Empty(harness.Text.Requests);
        Assert.Empty(speech.Synthesized);
        Assert.Contains(harness.Session.Transcript, t => t.Role == TextRole.User && t.Text == "open the mixer");
    }

    [Fact]
    public async Task ATranscribeOnlySessionRefusesATypedMessage()
    {
        await using VoiceHarness harness = await VoiceHarness.StartAsync(Options(transcribeOnly: true));

        await Assert.ThrowsAsync<InvalidOperationException>(() => harness.Session.SubmitUserTextAsync("hello"));
    }
}
