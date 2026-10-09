using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe.Residency;
using HartsyInference.Core.Moe.Telemetry;
using Xunit;

namespace HartsyInference.Core.Tests.Moe.Residency;

/// <summary>Residency policies: hot experts stay resident, replays are deterministic, and the hot path allocates nothing.</summary>
public sealed class ResidencyPolicyTests
{
    private const int SlotBudget = 4;

    public static TheoryData<string> PolicyNames => new() { "SegmentedLru", "Lfu", "DecayedLfuHysteresis" };

    /// <summary>Builds a fresh policy by name. Parameters are fixed so the tests are reproducible.</summary>
    internal static IAdaptiveResidencyPolicy Create(string name) => name switch
    {
        "SegmentedLru" => new SegmentedLruPolicy(protectedCapacity: 2),
        "Lfu" => new LfuPolicy(),
        "DecayedLfuHysteresis" => new DecayedLfuHysteresisPolicy(decayPerStep: 0.95, hysteresisMargin: 0.1),
        _ => throw new ArgumentException(name, nameof(name)),
    };

    [Theory]
    [MemberData(nameof(PolicyNames))]
    public void HotExpert_StaysResidentInASkewedTrace(string name)
    {
        // Expert 0 is routed on every other access; a cold scan over 50 other experts fills the rest.
        ExpertKey hot = new(0, 0);
        SlotCacheSimulator simulator = new(Create(name), SlotBudget);
        long hotMisses = 0;
        for (int i = 0; i < 1000; i++)
        {
            ExpertKey key = (i & 1) == 0 ? hot : new ExpertKey(0, 1 + ((i / 2) % 50));
            bool hit = simulator.Access(key, 1024);
            if (key == hot && !hit) hotMisses++;
        }

        Assert.Equal(1, hotMisses);
        Assert.Contains(hot, simulator.Residents.ToArray());
    }

    [Theory]
    [MemberData(nameof(PolicyNames))]
    public void Replay_IsDeterministic(string name)
    {
        ExpertTraceRecord[] trace = SyntheticTraceGenerator.Zipf(1234, 4, 32, 400, 2, 1.0, 1 << 20);

        ReplayResult first = TraceReplayHarness.Replay(trace, Create(name), 24);
        ReplayResult second = TraceReplayHarness.Replay(trace, Create(name), 24);

        Assert.Equal(first, second);
        Assert.Equal(first.Hits + first.Misses, first.Accesses);
        Assert.Equal(first.Hits, TraceReplayHarness.Replay(trace, WarmedThenReset(name, trace), 24).Hits);
    }

    [Theory]
    [MemberData(nameof(PolicyNames))]
    public void AccessAndChooseVictim_AllocateNothingAfterWarmUp(string name)
    {
        IAdaptiveResidencyPolicy policy = Create(name);
        ExpertKey[] keys = new ExpertKey[8];
        for (int i = 0; i < keys.Length; i++) keys[i] = new ExpertKey(0, i);
        ExpertKey[] resident = keys[..SlotBudget];

        RunCycle(policy, keys, resident, 2000);

        long before = GC.GetAllocatedBytesForCurrentThread();
        RunCycle(policy, keys, resident, 20000);
        long after = GC.GetAllocatedBytesForCurrentThread();

        Assert.Equal(0L, after - before);
    }

    [Fact]
    public void DecayedPolicy_ScoreHalvesPerStep()
    {
        DecayedLfuHysteresisPolicy policy = new(decayPerStep: 0.5, hysteresisMargin: 0.0);
        ExpertKey a = new(0, 0);
        ExpertKey b = new(0, 1);
        policy.NoteAccess(a);
        policy.NoteAccess(b);
        policy.NoteAccess(b);

        // a scored 1 at step 1 and has decayed twice by step 3; b scored 1 at step 2, decayed once, plus 1 at step 3.
        Assert.Equal(0.25, policy.ScoreOf(a), 6);
        Assert.Equal(1.5, policy.ScoreOf(b), 6);
    }

    [Fact]
    public void DecayedPolicy_HysteresisKeepsIncumbentUntilChallengerIsClearlyWeaker()
    {
        // No decay, so scores are plain counts. Margin 0.5: a challenger must fall below half the incumbent's score.
        DecayedLfuHysteresisPolicy policy = new(decayPerStep: 1.0, hysteresisMargin: 0.5);
        ExpertKey a = new(0, 0);
        ExpertKey c = new(0, 1);
        ExpertKey d = new(0, 2);
        ExpertKey e = new(0, 3);
        for (int i = 0; i < 4; i++) policy.NoteAccess(a);
        for (int i = 0; i < 2; i++) policy.NoteAccess(c);
        policy.NoteAccess(d);

        Assert.Equal(c, policy.ChooseVictim([c, a]));

        // d (1) is weaker than the incumbent c (2) but not below c * 0.5, so the incumbent stays the victim.
        Assert.Equal(c, policy.ChooseVictim([c, d, a]));

        // e has never been routed (score 0) and is below the margin, so the victim switches to it.
        Assert.Equal(e, policy.ChooseVictim([c, d, a, e]));
    }

    private static IAdaptiveResidencyPolicy WarmedThenReset(string name, ExpertTraceRecord[] trace)
    {
        IAdaptiveResidencyPolicy policy = Create(name);
        TraceReplayHarness.Replay(trace, policy, SlotBudget);
        policy.Reset();
        return policy;
    }

    private static void RunCycle(IAdaptiveResidencyPolicy policy, ExpertKey[] keys, ExpertKey[] resident, int iterations)
    {
        for (int i = 0; i < iterations; i++)
        {
            ExpertKey key = keys[i % keys.Length];
            policy.NoteAccess(key);
            if (Array.IndexOf(resident, key) >= 0) continue;

            ExpertKey victim = policy.ChooseVictim(resident);
            int at = Array.IndexOf(resident, victim);
            policy.NoteEvict(victim);
            resident[at] = key;
            policy.NoteInsert(key);
        }
    }
}
