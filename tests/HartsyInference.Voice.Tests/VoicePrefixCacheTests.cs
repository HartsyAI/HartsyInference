using HartsyInference.Engine.Requests;
using HartsyInference.Voice.Tests.Fakes;
using Xunit;

namespace HartsyInference.Voice.Tests;

/// <summary>Coverage for the voice-layer side of prefix-KV reuse: a per-call key that is stable across a call's
/// turns, a priming request fired at <see cref="VoiceAgentSession.StartAsync"/> so turn 1 is warm too, and the
/// opt-out. <see cref="VoiceHarness.DefaultOptions"/> turns this off for every OTHER test in this project (see its
/// own doc) so this file opts back in explicitly.</summary>
public sealed class VoicePrefixCacheTests
{
    private static VoiceAgentOptions Options() => VoiceHarness.DefaultOptions() with { EnablePrefixCache = true };

    [Fact]
    public async Task StartAsync_PrimesTheSystemAndToolsPrefix_BeforeAnyRealTurn()
    {
        ScriptedTextService text = new ScriptedTextService().Reply("Hello there.");
        await using VoiceHarness harness = await VoiceHarness.StartAsync(Options(), text: text);

        // StartAsync's priming call (ScriptedTextService.GenerateAsync completes synchronously, with no real
        // async gap) lands before any turn this test itself triggers.
        TextRequest priming = Assert.Single(text.Requests);
        Assert.Single(priming.Messages);
        Assert.Equal(TextRole.System, priming.Messages[0].Role);
        Assert.Equal(1, priming.MaxTokens);
        Assert.False(priming.EnableThinking);
        Assert.NotNull(priming.PrefixCacheKey);
        Assert.StartsWith("voice:", priming.PrefixCacheKey);
        // No capacity hint: the engine grows the retained KV by copy as the conversation needs it, so sizing the
        // priming request for the whole history would only allocate a buffer to shrink right after.
        Assert.Null(priming.PrefixCacheCapacityHint);

        harness.Session.PushDtmf('1');
        await harness.TurnCompletedAsync(1);

        Assert.Equal(2, text.Requests.Count);
        TextRequest[] requests = [.. text.Requests];
        TextRequest turn = requests[1];
        Assert.Equal(priming.PrefixCacheKey, turn.PrefixCacheKey);          // same call, same key
        Assert.Null(turn.PrefixCacheCapacityHint);
        Assert.True(turn.Messages.Count > 1, "a real turn's request carries more than just the system message.");
    }

    [Fact]
    public async Task EnablePrefixCache_False_SendsNoPrimingRequest_AndNoKeyOnRealTurns()
    {
        ScriptedTextService text = new ScriptedTextService().Reply("Hello there.");
        VoiceAgentOptions options = VoiceHarness.DefaultOptions() with { EnablePrefixCache = false };
        await using VoiceHarness harness = await VoiceHarness.StartAsync(options, text: text);

        Assert.Empty(text.Requests);   // nothing fired at start -- opted out

        harness.Session.PushDtmf('1');
        await harness.TurnCompletedAsync(1);

        TextRequest turn = Assert.Single(text.Requests);
        Assert.Null(turn.PrefixCacheKey);
        Assert.Null(turn.PrefixCacheCapacityHint);
    }
}
