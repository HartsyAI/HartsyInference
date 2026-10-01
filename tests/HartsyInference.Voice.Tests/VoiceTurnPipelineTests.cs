using HartsyInference.Audio.Models.Denoise;
using HartsyInference.Core.Logging;
using HartsyInference.Cpu;
using HartsyInference.Engine.Audio.Wake;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;
using HartsyInference.Tools;
using HartsyInference.Voice.Audio;
using HartsyInference.Voice.Tests.Fakes;
using HartsyInference.Voice.Turns;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Voice.Tests;

/// <summary>Whole turns through a started session on fake models: sentences play in order at contiguous offsets, a slow
/// reader loses nothing, a tool call is dispatched and the model re-invoked with its result, history is trimmed to the
/// budget, every language-model request goes out with thinking off on the LLM device, utterances are recognized on the
/// GPU thread, a revoked speech model is reopened once, barge-in stops the reply and the interrupting speech is answered,
/// and the per-turn metrics line carries every <c>voice.*</c> key.</summary>
public sealed class VoiceTurnPipelineTests
{
    private const int Sentence = 2_400;

    private readonly ITestOutputHelper _output;

    public VoiceTurnPipelineTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task SentencesPlayInOrderAtContiguousOffsets()
    {
        ScriptedTextService text = new ScriptedTextService().Reply("The first sentence is here. The second one follows it. And a third to end.");
        await using VoiceHarness harness = await VoiceHarness.StartAsync(text: text);
        harness.Session.PushDtmf('5');
        await harness.TurnCompletedAsync(1);

        Assert.Equal(["The first sentence is here.", "The second one follows it.", "And a third to end."], harness.Speech.Synthesized);
        float[] played = harness.Reader.Samples;
        Assert.Equal(3 * Sentence, played.Length);
        for (int sentence = 0; sentence < 3; sentence++)
        {
            float marker = FakeSpeech.Marker(sentence + 1);
            Assert.All(played.AsSpan(sentence * Sentence, Sentence).ToArray(), sample => Assert.Equal(marker, sample));
        }
        Assert.Equal("[DTMF 5]", Assert.Single(text.Requests).Messages[^1].Content);
    }

    [Fact]
    public async Task AReplyWithNoWordsIsNotSentToTheSynthesizer()
    {
        // The engine's synthesizer lease fails when the model yields no audio, which is what a lone "..." gives it.
        ScriptedTextService text = new ScriptedTextService().Reply("...").Reply("Sure, I can help with that.");
        await using VoiceHarness harness = await VoiceHarness.StartAsync(text: text);
        harness.Session.PushDtmf('6');
        await harness.TurnCompletedAsync(1);
        harness.Session.PushDtmf('7');
        await harness.TurnCompletedAsync(2);

        Assert.Equal(["Sure, I can help with that."], harness.Speech.Synthesized);
        Assert.DoesNotContain(harness.Events, e => e.Kind == VoiceAgentEventKind.Error);
        Assert.Equal(Sentence, harness.Reader.Samples.Length);
    }

    [Fact]
    public async Task ResampledPlaybackHasTheConvertedLength()
    {
        ScriptedTextService text = new ScriptedTextService().Reply("One second of reply audio.");
        VoiceAgentOptions options = VoiceHarness.DefaultOptions() with { OutboundSampleRate = 16_000 };
        await using VoiceHarness harness = await VoiceHarness.StartAsync(options, new FakeSpeech { SamplesPerSentence = 24_000 }, text);
        harness.Session.PushDtmf('1');
        await harness.TurnCompletedAsync(1);

        float[] played = harness.Reader.Samples;
        // 50 frames of 480 → 50 of 320 (the first is the resampler's lag), plus the padded tail and one flush frame.
        Assert.Equal(16_000 + 2 * 320, played.Length);
        float marker = FakeSpeech.Marker(1);
        Assert.All(played.AsSpan(1_000, 14_000).ToArray(), sample => Assert.InRange(sample, marker * 0.99f, marker * 1.01f));
    }

    [Fact]
    public async Task ASlowReaderLosesNothing()
    {
        ScriptedTextService text = new ScriptedTextService().Reply(
            "Sentence number one is here. Sentence number two is here. Sentence number three is here. Sentence number four is here.");
        await using VoiceHarness harness = await VoiceHarness.StartAsync(text: text, speech: new FakeSpeech { SamplesPerSentence = 4_800 },
            outboundCapacity: 1_024, readerPause: TimeSpan.FromMilliseconds(2));
        harness.Session.PushDtmf('2');
        await harness.TurnCompletedAsync(1, seconds: 60);

        float[] played = harness.Reader.Samples;
        Assert.Equal(4 * 4_800, played.Length);
        for (int sentence = 0; sentence < 4; sentence++)
        {
            Assert.All(played.AsSpan(sentence * 4_800, 4_800).ToArray(), sample => Assert.Equal(FakeSpeech.Marker(sentence + 1), sample));
        }
        Assert.Equal(0, harness.Session.InboundDroppedSamples);
    }

    [Fact]
    public async Task ReadingAReplyWhileItsTurnWaitsForPlaybackAllocatesNothing()
    {
        // The turn waits for playback with its cancellable token; the reader used to pay 32 B per read for that wait.
        ScriptedTextService text = new ScriptedTextService().Reply("A first reply warms the reader up.")
            .Reply("The measured reply plays for a while. It has a second sentence. And a third one to end.");
        await using VoiceHarness harness = await VoiceHarness.StartAsync(text: text, speech: new FakeSpeech { SamplesPerSentence = 24_000 });
        harness.Session.PushDtmf('1');
        await harness.TurnCompletedAsync(1);
        long allocated = harness.Reader.ReadAllocatedBytes;
        int reads = harness.Reader.Reads.Count;

        harness.Session.PushDtmf('2');
        await harness.TurnCompletedAsync(2);
        int measured = harness.Reader.Reads.Count - reads;
        Assert.True(measured >= 3 * 24_000 / 320, $"only {measured} reads returned reply audio.");
        Assert.Equal(allocated, harness.Reader.ReadAllocatedBytes);
    }

    [Fact]
    public async Task AToolCallRunsAndTheModelIsAskedAgainWithItsResult()
    {
        int invoked = 0;
        ToolRegistry tools = new ToolRegistry().Add("get_time", "Tells the time.", "{\"type\":\"object\"}", (_, _) =>
        {
            invoked++;
            return Task.FromResult("12:30");
        });
        ScriptedTextService text = new ScriptedTextService()
            .Round(ScriptedTextService.Text("Let me check the time. "), ScriptedTextService.Call("call_0", "get_time"),
                ScriptedTextService.Stop(StopReason.ToolCall))
            .Reply("It is half past twelve.")
            .Reply("You are welcome, goodbye.");
        await using VoiceHarness harness = await VoiceHarness.StartAsync(text: text, tools: tools);
        harness.Session.PushDtmf('1');
        await harness.TurnCompletedAsync(1);

        Assert.Equal(1, invoked);
        TextRequest[] requests = [.. text.Requests];
        Assert.Equal(2, requests.Length);
        Assert.Same(tools.Definitions, requests[0].Tools);
        IReadOnlyList<TextMessage> second = requests[1].Messages;
        Assert.Equal([TextRole.System, TextRole.User, TextRole.Assistant, TextRole.Tool], second.Select(m => m.Role));
        Assert.Equal("get_time", Assert.Single(second[2].ToolCalls!).Name);
        Assert.Equal("12:30", second[3].Content);
        Assert.Equal("call_0", second[3].ToolCallId);
        Assert.Equal(["Let me check the time.", "It is half past twelve."], harness.Speech.Synthesized);
        Assert.Contains(harness.Events, e => e.Kind == VoiceAgentEventKind.StateChanged && e.State == VoiceAgentState.ToolRunning);
        Assert.Contains(harness.Events, e => e.Kind == VoiceAgentEventKind.ToolResult && e.Text == "12:30" && e.ToolCall!.Name == "get_time");
        VoiceTurnMetrics metrics = (await harness.TurnCompletedAsync(1)).Metrics!.Value;
        Assert.Equal(1, metrics.ToolCalls);

        harness.Session.PushDtmf('2');
        await harness.TurnCompletedAsync(2);
        IReadOnlyList<TextMessage> third = text.Requests.Last().Messages;
        Assert.Equal([TextRole.System, TextRole.User, TextRole.Assistant, TextRole.Tool, TextRole.Assistant, TextRole.User],
            third.Select(m => m.Role));
        Assert.Equal("It is half past twelve.", third[4].Content);
    }

    [Fact]
    public async Task HistoryIsTrimmedToTheTokenBudget()
    {
        ScriptedTextService text = new();
        for (int turn = 0; turn < 8; turn++)
        {
            text.Reply($"Reply number {turn} has six words.");
        }
        VoiceAgentOptions options = VoiceHarness.DefaultOptions() with { SystemPrompt = "Be brief.", MaxHistoryTokens = 50 };
        await using VoiceHarness harness = await VoiceHarness.StartAsync(options, text: text);
        for (int turn = 1; turn <= 8; turn++)
        {
            harness.Session.PushDtmf((char)('0' + turn));
            await harness.TurnCompletedAsync(turn);
        }

        IReadOnlyList<TextMessage> last = text.Requests.Last().Messages;
        Assert.Equal("Be brief.", last[0].Content);
        Assert.Equal(TextRole.User, last[1].Role);
        Assert.Equal("[DTMF 8]", last[^1].Content);
        int tokens = last.Sum(m => m.Content.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length + 4);
        Assert.InRange(tokens, 1, 50);
        Assert.True(last.Count < 16, $"history was not trimmed: {last.Count} messages.");
    }

    [Fact]
    public async Task EveryModelRequestHasThinkingOffOnTheLlmDevice()
    {
        ToolRegistry tools = new ToolRegistry().Add("hang_up", "Ends the call.", "{\"type\":\"object\"}", (_, _) => Task.FromResult("ok"));
        ScriptedTextService text = new ScriptedTextService()
            .Round(ScriptedTextService.Call("call_0", "hang_up"), ScriptedTextService.Stop(StopReason.ToolCall))
            .Reply("Goodbye then.")
            .Reply("Sure thing, anything else?");
        VoiceAgentOptions options = VoiceHarness.DefaultOptions() with { LlmDevice = "cuda:0", MaxReplyTokens = 77 };
        await using VoiceHarness harness = await VoiceHarness.StartAsync(options, text: text, tools: tools);
        await harness.Models.WarmAsync(text);
        harness.Session.PushDtmf('9');
        await harness.TurnCompletedAsync(1);
        harness.Speech.Transcripts.Enqueue("is there anything else");
        harness.PushSilence(0.3);
        harness.PushSpeech(1.0);
        harness.PushSilence(1.0);
        await harness.TurnCompletedAsync(2);

        TextRequest[] requests = [.. text.Requests];
        Assert.Equal(4, requests.Length);
        Assert.All(requests, request =>
        {
            Assert.False(request.EnableThinking);
            Assert.Equal("cuda:0", request.Device);
            Assert.False(request.AlwaysFreeMemory);
        });
        Assert.All(requests.Where(r => r.MaxTokens != 1), request =>
        {
            Assert.Equal(77, request.MaxTokens);
            Assert.Same(tools.Definitions, request.Tools);
        });
    }

    [Fact]
    public async Task AnUtteranceIsRecognizedOnTheGpuThreadAndAnsweredWithEveryMetricLogged()
    {
        List<string> log = [];
        Logs.SetLogger((_, message) =>
        {
            lock (log)
            {
                log.Add(message);
            }
        });
        try
        {
            ScriptedTextService text = new ScriptedTextService().Reply("It is noon.");
            await using VoiceHarness harness = await VoiceHarness.StartAsync(text: text);
            harness.Speech.Transcripts.Enqueue(" What time is it? ");
            harness.PushSilence(0.3);
            harness.PushSpeech(1.0);
            harness.PushSilence(1.0);
            VoiceAgentEvent done = await harness.TurnCompletedAsync(1);

            Assert.Equal("What time is it?", (await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.UserTranscript)).Text);
            Assert.Equal("What time is it?", text.Requests.Single().Messages[^1].Content);
            Assert.All(harness.Speech.Threads, thread => Assert.Equal(harness.Models.Gpu.ManagedThreadId, thread));
            int heard = Assert.Single(harness.Speech.TranscribedSamples);
            Assert.InRange(heard, 16_000, 16_000 + 2 * 480 + 512);

            VoiceTurnMetrics metrics = done.Metrics!.Value;
            Assert.Equal(VoiceTurnKind.Utterance, metrics.Kind);
            Assert.InRange(metrics.EndpointMs!.Value, 700, 764);
            Assert.NotNull(metrics.SttMs);
            Assert.NotNull(metrics.LlmTtftMs);
            Assert.NotNull(metrics.LlmFirstSentenceMs);
            Assert.NotNull(metrics.TtsFirstChunkMs);
            Assert.NotNull(metrics.TransportMs);
            Assert.NotNull(metrics.FrontendMaxMs);
            Assert.True(metrics.TotalMs >= metrics.EndpointMs);
            Assert.False(metrics.Interrupted);

            string line;
            lock (log)
            {
                line = Assert.Single(log, message => message.StartsWith("[Voice] turn 1 (utterance):", StringComparison.Ordinal));
            }
            foreach (string key in VoiceTurnMetrics.MetricKeys)
            {
                Assert.Contains(" " + key + "=", line, StringComparison.Ordinal);
            }
            Assert.Equal([new VoiceTranscriptEntry(1, TextRole.User, "What time is it?", false), new VoiceTranscriptEntry(1, TextRole.Assistant, "It is noon.", false)],
                harness.Session.Transcript);
        }
        finally
        {
            Logs.SetLogger(null!);
        }
    }

    [Fact]
    public void FrontendDenoiserLatencySamplesReflectsTheRealDenoisersAlgorithmicLagOrZeroWithoutOne()
    {
        // A real RNNoise instance ahead of the fake level-scripted VAD would make this a session-level test instead
        // (push a turn, read VoiceTurnMetrics.EndpointMs), but RNNoise legitimately suppresses a constant-level tone
        // as non-speech noise (confirmed: with Denoise on, LevelVadModel never sees speech and the turn never ends),
        // so that combination cannot drive a turn at all. This checks the one new property Turn.Metrics() reads
        // (VoiceAgentSession.Turns.cs) directly: VoiceAudioFrontend.DenoiserLatencySamples. The arithmetic that adds
        // it into EndpointMs/TotalMs is otherwise covered by AnUtteranceIsRecognizedOnTheGpuThreadAndAnsweredWithEveryMetricLogged
        // above, which proves the unchanged (Denoise off, latency 0) case still lands in its established [700, 764] range.
        if (!RealWeightGate.Require(_output.WriteLine, VoiceAssets.RnnoiseWeights, VoiceAssets.RnnoiseInt8Tables))
        {
            return;
        }
        using WakeModelSet wake = new(VoiceAssets.WakeRoot);
        Assert.True(wake.LoadDenoiser(RnnoisePrecision.Int8));
        RnnoiseStream denoiser = wake.CreateDenoiser() ?? throw new InvalidOperationException("Could not instantiate RNNoise.");
        using CpuBackend cpu = new();
        VoiceTurnSignals signals = new();
        VoiceAgentOptions options = new();

        using VoiceAudioFrontend withDenoiser = new(cpu, new LevelVadModel(), denoiser, signals, options);
        _output.WriteLine($"real RNNoise LatencySamples = {denoiser.LatencySamples} ({denoiser.LatencySamples / 16.0:F1} ms at 16 kHz)");
        Assert.True(denoiser.LatencySamples > 0, "a real denoiser should report a non-zero algorithmic lag.");
        Assert.Equal(denoiser.LatencySamples, withDenoiser.DenoiserLatencySamples);

        using VoiceAudioFrontend withoutDenoiser = new(cpu, new LevelVadModel(), null, signals, options);
        Assert.Equal(0, withoutDenoiser.DenoiserLatencySamples);
    }

    [Fact]
    public async Task AnUtteranceWithNoWordsIsNotAnswered()
    {
        ScriptedTextService text = new();
        await using VoiceHarness harness = await VoiceHarness.StartAsync(text: text);
        harness.Speech.Transcripts.Enqueue(" . ");
        harness.PushSilence(0.3);
        harness.PushSpeech(0.6);
        harness.PushSilence(1.0);
        await harness.TurnCompletedAsync(1);

        Assert.Empty(text.Requests);
        Assert.Equal(1, harness.Session.DiscardedUtterances);
        Assert.Contains(harness.Events, e => e.Kind == VoiceAgentEventKind.UtteranceDiscarded && e.TurnId == 1);
        Assert.Equal(VoiceAgentState.Listening, harness.Session.State);
    }

    [Fact]
    public async Task SpeakAsyncSaysTheTextWithoutTheModelAndItJoinsTheConversation()
    {
        ScriptedTextService text = new ScriptedTextService().Reply("Happy to help.");
        await using VoiceHarness harness = await VoiceHarness.StartAsync(text: text);
        await harness.Session.SpeakAsync("Hello, thanks for calling. How can I help?").WaitAsync(TimeSpan.FromSeconds(20));

        Assert.Empty(text.Requests);
        Assert.Equal(2, harness.Speech.Synthesized.Count);
        harness.Session.PushDtmf('0');
        await harness.TurnCompletedAsync(2);
        IReadOnlyList<TextMessage> messages = text.Requests.Single().Messages;
        Assert.Equal([TextRole.System, TextRole.Assistant, TextRole.User], messages.Select(m => m.Role));
        Assert.Equal("Hello, thanks for calling. How can I help?", messages[1].Content);
    }

    [Fact]
    public async Task ARevokedSpeechModelIsReopenedOnceAndTheTurnCompletes()
    {
        ScriptedTextService text = new ScriptedTextService().Reply("Still here and talking.");
        await using VoiceHarness harness = await VoiceHarness.StartAsync(text: text);
        harness.Speech.Revoked = true;
        harness.Session.PushDtmf('3');
        await harness.TurnCompletedAsync(1);

        Assert.Equal(1, harness.Speech.Reopens);
        Assert.Equal(1, harness.Models.Gpu.Reopens);
        Assert.Equal(Sentence, harness.Reader.Samples.Length);
        Assert.DoesNotContain(harness.Events, e => e.Kind == VoiceAgentEventKind.Error);
    }

    [Fact]
    public async Task AReopenThatFailsEndsTheTurnWithAnErrorAndKeepsListening()
    {
        ScriptedTextService text = new ScriptedTextService().Reply("This will not be heard.").Reply("Back again now.");
        await using VoiceHarness harness = await VoiceHarness.StartAsync(text: text);
        harness.Speech.Revoked = true;
        harness.Speech.ReopenFailure = new InvalidOperationException("the engine is gone");
        harness.Session.PushDtmf('4');
        await harness.TurnCompletedAsync(1);

        VoiceAgentEvent error = await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.Error);
        Assert.Equal(1, error.TurnId);
        Assert.Contains("the engine is gone", error.Text, StringComparison.Ordinal);
        await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.StateChanged && e.State == VoiceAgentState.Listening && e.TurnId == 1);

        harness.Speech.ReopenFailure = null;
        harness.Session.PushDtmf('5');
        await harness.TurnCompletedAsync(2);
        Assert.Equal(Sentence, harness.Reader.Samples.Length);
    }

    [Fact]
    public async Task BargeInStopsTheReplyAndTheInterruptingSpeechIsAnswered()
    {
        ScriptedTextService text = new ScriptedTextService()
            .Reply("This is a long answer. It goes on for quite a while. There is much more to say here. And still more after that.")
            .Reply("Okay, stopping.");
        // Four seconds of reply per sentence, read at playback speed, so the caller has time to talk over it.
        await using VoiceHarness harness = await VoiceHarness.StartAsync(text: text, speech: new FakeSpeech { SamplesPerSentence = 96_000 },
            readerPause: TimeSpan.FromMilliseconds(20), startReader: true);
        harness.Session.PushDtmf('1');
        await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.StateChanged && e.State == VoiceAgentState.Speaking);
        await harness.Reader.WaitForSamplesAsync(4_800);

        harness.Speech.Transcripts.Enqueue("stop please");
        harness.PushSilence(0.4);
        harness.PushSpeech(0.6);
        VoiceAgentEvent bargeIn = await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.BargeIn);
        Assert.Equal(1, bargeIn.TurnId);
        VoiceTurnMetrics interrupted = (await harness.TurnCompletedAsync(1)).Metrics!.Value;
        Assert.True(interrupted.Interrupted);
        Assert.NotNull(interrupted.BargeInStopMs);
        int playedWhenStopped = harness.Reader.Samples.Length;
        Assert.True(playedWhenStopped < 4 * 96_000, "the whole reply played despite the barge-in.");

        harness.PushSilence(1.0);
        await harness.TurnCompletedAsync(2);
        Assert.Equal("stop please", text.Requests.Last().Messages[^1].Content);
        VoiceTranscriptEntry reply = harness.Session.Transcript.First(e => e.TurnId == 1 && e.Role == TextRole.Assistant);
        Assert.True(reply.Interrupted);
        // Nothing from the interrupted reply after the flush: every later sample is turn 2's marker.
        float[] played = harness.Reader.Samples;
        float firstOfTurnTwo = FakeSpeech.Marker(harness.Speech.Synthesized.Count);
        int turnTwoStart = Array.IndexOf(played, firstOfTurnTwo);
        Assert.True(turnTwoStart >= 0);
        Assert.All(played.AsSpan(turnTwoStart).ToArray(), sample => Assert.Equal(firstOfTurnTwo, sample));
    }

    [Fact]
    public async Task TheDevicePoolIsTrimmedOnceWhenTheSessionReturnsToListening()
    {
        ScriptedTextService text = new ScriptedTextService().Reply("One short reply.");
        await using VoiceHarness harness = await VoiceHarness.StartAsync(text: text);
        Assert.Equal(0, harness.Models.Gpu.Trims);
        harness.Session.PushDtmf('8');
        await harness.TurnCompletedAsync(1);
        await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.StateChanged && e.State == VoiceAgentState.Listening && e.TurnId == 1);

        // No further GPU job is queued: the trim runs on its own, between turns.
        DateTime deadline = DateTime.UtcNow.AddSeconds(5);
        while (harness.Models.Gpu.Trims == 0 && DateTime.UtcNow < deadline)
        {
            await Task.Delay(5);
        }
        Assert.Equal(1, harness.Models.Gpu.Trims);
    }

    [Fact]
    public async Task EveryJobKeepsThePoolAndEachTurnTrimsItOnceOnItsReturnToListening()
    {
        ScriptedTextService text = new ScriptedTextService().Reply("The first reply has one sentence.")
            .Reply("The second reply has two sentences. Here is the other one.");
        RecordingDevice? recorder = null;
        await using VoiceHarness harness = await VoiceHarness.StartAsync(text: text, device: cpu => (recorder = RecordingDevice.Wrap(cpu)).Backend);
        for (int turn = 1; turn <= 2; turn++)
        {
            harness.Session.PushDtmf('1');
            await harness.TurnCompletedAsync(turn);
            int id = turn;
            await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.StateChanged && e.State == VoiceAgentState.Listening && e.TurnId == id);
            DateTime deadline = DateTime.UtcNow.AddSeconds(5);
            while (harness.Models.Gpu.Trims < turn && DateTime.UtcNow < deadline)
            {
                await Task.Delay(5);
            }
        }
        await Task.Delay(100);

        // One synthesis per sentence, each freeing its activations with the pool kept, then one trim per turn.
        string free = RecordingDevice.FreeKeepingPool;
        string trim = RecordingDevice.Trim;
        Assert.Equal([free, trim, free, free, trim], recorder!.Calls);
        Assert.Equal(2, harness.Models.Gpu.Trims);
    }

    [Fact]
    public async Task ACancelledStartEndsTheSession()
    {
        using CpuBackend device = new();
        VoiceAgentOptions options = VoiceHarness.DefaultOptions();
        await using VoiceModelSet models = new(options, new FakeSpeech(), device, () => new LevelVadModel(), createDenoiser: null);
        VoiceAgentSession session = new(models, new ScriptedTextService(), new ToolRegistry(), options);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => session.StartAsync(new CancellationToken(canceled: true)));

        Assert.Equal(VoiceAgentState.Ended, session.State);
        Assert.Throws<InvalidOperationException>(() => session.PushDtmf('1'));
        await session.EndAsync().WaitAsync(TimeSpan.FromSeconds(10));
    }

    [Fact]
    public async Task EndingCancelsQueuedPromptsAndEndsTheSession()
    {
        using ManualResetEventSlim hold = new(false);
        await using VoiceHarness harness = await VoiceHarness.StartAsync(speech: new FakeSpeech { HoldSynthesis = hold });
        Task first = harness.Session.SpeakAsync("The first prompt is held on the GPU.");
        Task second = harness.Session.SpeakAsync("The second prompt never starts.");
        await harness.WaitForAsync(e => e.Kind == VoiceAgentEventKind.StateChanged && e.State == VoiceAgentState.Thinking);

        Task ending = harness.Session.EndAsync();
        hold.Set();
        await ending.WaitAsync(TimeSpan.FromSeconds(20));

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => first.WaitAsync(TimeSpan.FromSeconds(5)));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => second.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Equal(VoiceAgentState.Ended, harness.Session.State);
        Assert.Throws<InvalidOperationException>(() => harness.Session.PushDtmf('1'));
    }
}
