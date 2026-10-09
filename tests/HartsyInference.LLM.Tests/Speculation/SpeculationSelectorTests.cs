using HartsyInference.LLM.Generation.Speculation;
using Xunit;

namespace HartsyInference.LLM.Tests.Speculation;

/// <summary>Drives <see cref="SpeculationSelector"/> with injected measurements only. No test reads the wall clock.</summary>
public sealed class SpeculationSelectorTests
{
    private static readonly SpeculationRound Rejected = new(Proposed: 4, Accepted: 0, TokensEmitted: 1, ElapsedMilliseconds: 1.0);
    private static readonly SpeculationRound Accepted = new(Proposed: 4, Accepted: 4, TokensEmitted: 5, ElapsedMilliseconds: 1.0);
    private static readonly SpeculationRound NoProposal = new(Proposed: 0, Accepted: 0, TokensEmitted: 1, ElapsedMilliseconds: 1.0);

    [Fact]
    public void ProviderWithLowAcceptance_IsDisabledExactlyAfterNRounds()
    {
        StubProvider low = new("low");
        StubProvider good = new("good");
        SpeculationSelector selector = new([low, good], new SpeculationSelectorOptions { DisableAfterRounds = 3 });

        selector.Record(low, Rejected);
        selector.Record(low, Rejected);
        Assert.False(selector.IsDisabled(low));

        selector.Record(low, Rejected);
        Assert.True(selector.IsDisabled(low));
        Assert.Same(good, selector.Select());
    }

    [Fact]
    public void AcceptedRoundAtThreshold_ResetsTheStreak()
    {
        StubProvider p = new("p");
        SpeculationSelector selector = new([p], new SpeculationSelectorOptions { DisableAfterRounds = 3, AcceptanceThreshold = 0.5 });

        selector.Record(p, Rejected);
        selector.Record(p, Rejected);
        // 2 of 4 is exactly the threshold, which is not below it, so the streak resets.
        selector.Record(p, new SpeculationRound(4, 2, 3, 1.0));
        selector.Record(p, Rejected);
        selector.Record(p, Rejected);
        Assert.False(selector.IsDisabled(p));

        selector.Record(p, Rejected);
        Assert.True(selector.IsDisabled(p));
    }

    [Fact]
    public void RoundsWithNoProposal_LeaveTheStreakUnchanged()
    {
        StubProvider p = new("p");
        SpeculationSelector selector = new([p], new SpeculationSelectorOptions { DisableAfterRounds = 3 });

        selector.Record(p, Rejected);
        selector.Record(p, Rejected);
        for (int i = 0; i < 5; i++) selector.Record(p, NoProposal);
        Assert.False(selector.IsDisabled(p));

        selector.Record(p, Rejected);
        Assert.True(selector.IsDisabled(p));
    }

    [Fact]
    public void FasterProvider_IsPreferredOnceBothAreMeasured()
    {
        StubProvider slow = new("slow");
        StubProvider fast = new("fast");
        SpeculationSelector selector = new([slow, fast]);

        // Unmeasured providers are tried first, in registration order.
        Assert.Same(slow, selector.Select());
        selector.Record(slow, new SpeculationRound(4, 4, 5, 10.0));   // 0.5 tokens per ms
        Assert.Same(fast, selector.Select());
        selector.Record(fast, new SpeculationRound(4, 4, 5, 2.0));    // 2.5 tokens per ms
        Assert.Same(fast, selector.Select());
    }

    [Fact]
    public void ThroughputMovesWithTheMovingAverage()
    {
        StubProvider p = new("p");
        SpeculationSelector selector = new([p], new SpeculationSelectorOptions { ThroughputSmoothing = 0.5 });

        selector.Record(p, new SpeculationRound(0, 0, 1, 1.0));   // first sample: 1.0 tokens per ms
        Assert.Equal(1.0, selector.Throughput(p)!.Value, 9);
        selector.Record(p, new SpeculationRound(0, 0, 3, 1.0));   // 1.0 + 0.5 * (3.0 - 1.0)
        Assert.Equal(2.0, selector.Throughput(p)!.Value, 9);
    }

    [Fact]
    public void SameMeasurements_GiveTheSameDecisions()
    {
        SpeculationRound[] script =
        [
            new(4, 0, 1, 2.0), new(4, 4, 5, 3.0), new(0, 0, 1, 1.0), new(4, 1, 2, 2.5),
            new(4, 0, 1, 2.0), new(4, 3, 4, 4.0), new(4, 0, 1, 2.0), new(4, 0, 1, 2.0),
        ];
        List<string> first = Replay(script);
        List<string> second = Replay(script);
        Assert.Equal(first, second);
        // "a" is measured first and is slower (0.5 tokens per ms) than "b" from its first timed round on,
        // so "b" is chosen after the first round.
        // Replay makes the choice before each round; the scripted rejections never reach the streak of 3 for "b".
        Assert.Equal(new[] { "a", "b", "b", "b", "b", "b", "b", "b" }, first);
    }

    [Fact]
    public void AllProvidersDisabled_SelectReturnsNull()
    {
        StubProvider only = new("only");
        SpeculationSelector selector = new([only], new SpeculationSelectorOptions { DisableAfterRounds = 1 });
        selector.Record(only, Rejected);
        Assert.Null(selector.Select());
    }

    [Fact]
    public void RecordingAnUnregisteredProvider_Throws()
    {
        SpeculationSelector selector = new([new StubProvider("a")]);
        Assert.Throws<ArgumentException>(() => selector.Record(new StubProvider("b"), Rejected));
    }

    [Fact]
    public void DuplicateProviderNames_AreRejected()
    {
        Assert.Throws<ArgumentException>(() => new SpeculationSelector([new StubProvider("x"), new StubProvider("x")]));
    }

    [Fact]
    public void InvalidRoundMeasurements_AreRejected()
    {
        StubProvider p = new("p");
        SpeculationSelector selector = new([p]);
        Assert.Throws<ArgumentOutOfRangeException>(() => selector.Record(p, new SpeculationRound(2, 3, 1, 1.0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => selector.Record(p, new SpeculationRound(2, 1, 1, -1.0)));
    }

    [Fact]
    public void InvalidOptions_AreRejected()
    {
        StubProvider p = new("p");
        Assert.Throws<ArgumentOutOfRangeException>(() => new SpeculationSelector([p], new SpeculationSelectorOptions { AcceptanceThreshold = 1.5 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SpeculationSelector([p], new SpeculationSelectorOptions { DisableAfterRounds = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SpeculationSelector([p], new SpeculationSelectorOptions { ThroughputSmoothing = 0.0 }));
    }

    /// <summary>Runs a fixed script with providers "a" and "b", and returns the selection made before each round.</summary>
    private static List<string> Replay(SpeculationRound[] script)
    {
        StubProvider a = new("a");
        StubProvider b = new("b");
        SpeculationSelector selector = new([a, b], new SpeculationSelectorOptions { DisableAfterRounds = 3 });
        List<string> choices = [];
        foreach (SpeculationRound round in script)
        {
            ISpeculativeDraftProvider? chosen = selector.Select();
            if (chosen is null)
            {
                choices.Add("none");
                continue;
            }
            choices.Add(chosen.Name);
            selector.Record(chosen, round);
        }
        return choices;
    }

    private sealed class StubProvider : ISpeculativeDraftProvider
    {
        public StubProvider(string name) => Name = name;

        public string Name { get; }

        public int[] Propose(int[] promptIds, IReadOnlyList<int> generated, int maxDraftLen) => [];
    }
}
