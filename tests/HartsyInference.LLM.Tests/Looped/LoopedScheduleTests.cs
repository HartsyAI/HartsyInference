using HartsyInference.LLM.Looped;
using Xunit;

namespace HartsyInference.LLM.Tests.Looped;

/// <summary>The unrolled layer order and cache-slot assignment of a looped decoder: slots must be stable across loop counts (a cache sized for the
/// maximum serves every smaller count), per-loop mode gives each pass its own recurrent slots, last-loop mode shares them.</summary>
public sealed class LoopedScheduleTests
{
    private static LoopedTransformerConfig Config(LoopedKvCacheMode mode, int loops = 2, int maxLoops = 0) => new()
    {
        HiddenSize = 16, NumHeads = 2, NumKvHeads = 1, HeadDim = 8, IntermediateSize = 24, VocabSize = 20,
        InputLayers = 2, RecurrentLayers = 3, OutputLayers = 2, Loops = loops, MaxLoops = maxLoops,
        LayerKinds = [LoopedLayerKind.SlidingAttention, LoopedLayerKind.Mlp,
            LoopedLayerKind.FullAttention, LoopedLayerKind.Mlp, LoopedLayerKind.SlidingAttention,
            LoopedLayerKind.FullAttention, LoopedLayerKind.Mlp],
        RopeThetas = [10_000f, 0f, 100_000f, 0f, 10_000f, 100_000f, 0f],
        SlidingWindow = 4, KvCacheMode = mode,
    };

    [Fact]
    public void PerLoop_UnrollsEveryPassWithItsOwnRecurrentSlots()
    {
        LoopedTransformerConfig cfg = Config(LoopedKvCacheMode.PerLoop);
        LoopedSchedule two = new(cfg, 2);

        Assert.Equal(2 + 3 * 2 + 2, two.VirtualLayers);
        Assert.Equal([0, 1, 2, 3, 4, 2, 3, 4, 5, 6], Enumerable.Range(0, two.VirtualLayers).Select(two.Physical));
        // input attention 0, output attention 1, then two recurrent groups of two attention layers each.
        Assert.Equal([0, -1, 2, -1, 3, 4, -1, 5, 1, -1], Enumerable.Range(0, two.VirtualLayers).Select(two.CacheSlot));
        Assert.Equal(6, two.SlotsUsed);
        Assert.Equal(2, two.RecurrentStart);
        Assert.Equal(8, two.RecurrentEnd);
    }

    [Fact]
    public void PerLoop_SlotsAreStableAcrossLoopCounts()
    {
        LoopedTransformerConfig cfg = Config(LoopedKvCacheMode.PerLoop);
        LoopedSchedule one = new(cfg, 1);
        LoopedSchedule four = new(cfg, 4);

        // The first pass and the output block use the same slots whatever the loop count.
        Assert.Equal([0, -1, 2, -1, 3, 1, -1], Enumerable.Range(0, one.VirtualLayers).Select(one.CacheSlot));
        Assert.Equal(0, four.CacheSlot(0));
        Assert.Equal(2, four.CacheSlot(2));
        Assert.Equal(3, four.CacheSlot(4));
        Assert.Equal(1, four.CacheSlot(four.VirtualLayers - 2));
        Assert.Equal(2 + 2 * 4, four.SlotsUsed);
        Assert.Equal(four.SlotsUsed, LoopedSchedule.MaxSlotCount(cfg));
        Assert.Equal(4, cfg.EffectiveMaxLoops);
    }

    [Fact]
    public void LastLoop_SharesTheRecurrentSlotsAcrossPasses()
    {
        LoopedTransformerConfig cfg = Config(LoopedKvCacheMode.LastLoop);
        LoopedSchedule two = new(cfg, 2);

        Assert.Equal([0, -1, 2, -1, 3, 2, -1, 3, 1, -1], Enumerable.Range(0, two.VirtualLayers).Select(two.CacheSlot));
        Assert.Equal(4, two.SlotsUsed);
        Assert.Equal(4, LoopedSchedule.MaxSlotCount(cfg));
    }

    [Fact]
    public void LoopBoundaries_MarkTheFirstAndLastRecurrentLayerOfEveryPass()
    {
        LoopedSchedule two = new(Config(LoopedKvCacheMode.PerLoop), 2);

        Assert.Equal([2, 5], Enumerable.Range(0, two.VirtualLayers).Where(two.IsLoopStart));
        Assert.Equal([4, 7], Enumerable.Range(0, two.VirtualLayers).Where(two.IsLoopEnd));
    }

    [Fact]
    public void LoopsOutsideTheModelRange_AreRefused()
    {
        LoopedTransformerConfig cfg = Config(LoopedKvCacheMode.PerLoop, loops: 2, maxLoops: 3);

        Assert.Throws<ArgumentOutOfRangeException>(() => new LoopedSchedule(cfg, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LoopedSchedule(cfg, 4));
        Assert.Equal(3, new LoopedSchedule(cfg, 3).Loops);
    }

    [Fact]
    public void Config_RejectsInconsistentShapes()
    {
        Assert.Throws<ArgumentException>(() => (Config(LoopedKvCacheMode.PerLoop) with { LayerKinds = [LoopedLayerKind.Mlp] }).Validate());
        Assert.Throws<ArgumentException>(() => (Config(LoopedKvCacheMode.PerLoop) with { SlidingWindow = 0 }).Validate());
        Assert.Throws<ArgumentException>(() => (Config(LoopedKvCacheMode.PerLoop) with { Loops = 0 }).Validate());
        Assert.Throws<ArgumentException>(() => (Config(LoopedKvCacheMode.PerLoop) with { NumKvHeads = 3 }).Validate());
        Config(LoopedKvCacheMode.PerLoop).Validate();
    }
}
