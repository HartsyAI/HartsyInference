using HartsyInference.Core.Backends;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The later run's budget at the two measured sizes: the reserved peak measured on the GPU, plus the safety margin, plus the cached step. The
/// resident blocks come from the planner's own sizing (ResidentPrefixSizing), with each block taken at 320 MiB. That block size is inferred: the
/// 640x368 measurement held seven resident blocks in 2443 MiB of spare, which bounds a block at 349 MiB, and earlier logs put them at 313-320 MiB.</summary>
public sealed class LongCatMeasuredLaterRunTests
{
    private const long Mb = 1L << 20;
    private static readonly long[] Blocks = Enumerable.Repeat(320 * Mb, 48).ToArray();

    private static LongCatConfig Config() => new() { HiddenSize = 4096, Depth = 48, PatchSize = (1, 2, 2) };

    /// <summary>The later run's noisy tokens: the new latents of a segment.</summary>
    private static long NoisyTokens(int width, int height)
    {
        LongCatAvatarChainPlan plan = LongCatAvatarChainPlan.For(24, 241);
        return LongCatRecipeSupport.TokenCount(plan.NewLatentsPerSegment, height / 8, width / 8);
    }

    private static long Cached(int width, int height) => LongCatRecipeSupport.CachedStepBytes(Config(), 5, height / 8, width / 8);

    /// <summary>The measured workload's key: the plan the measurement ran (24-latent segments, 4 context latents, 20 new, distilled).</summary>
    private static long? Peak(int width, int height, int newLatents = 20, int contextLatents = 4, bool distilled = true) =>
        LongCatRecipeSupport.MeasuredReservedPeak(DeviceType.Cuda, width, height, newLatents, contextLatents, distilled);

    private static long Measured(int width, int height, long cache) =>
        LongCatRecipeSupport.BudgetHeadroomBytes(NoisyTokens(width, height), 4096, 11008, Peak(width, height), cache);

    private static long Estimate(int width, int height, long cache) =>
        LongCatRecipeSupport.BudgetHeadroomBytes(NoisyTokens(width, height), 4096, 11008, measuredPeakBytes: null, cacheBytes: cache);

    [Fact]
    public void OnlyTheMeasuredWorkloadsHaveAPeak()
    {
        Assert.Equal(7808 * Mb, Peak(640, 368));
        Assert.Equal(11360 * Mb, Peak(832, 480));
        Assert.Null(Peak(1024, 576));
        Assert.Null(Peak(832, 464));
        Assert.Null(Peak(832, 480, distilled: false));
    }

    [Fact]
    public void AVulkanRunHasNoMeasuredPeakEvenAtAMeasuredSize()
    {
        // The measurement is CUDA's reserved bytes; the same size on Vulkan keeps the estimate.
        Assert.Null(LongCatRecipeSupport.MeasuredReservedPeak(DeviceType.Vulkan, 640, 368, 20, 4, true));
        Assert.Null(LongCatRecipeSupport.MeasuredReservedPeak(DeviceType.Vulkan, 832, 480, 20, 4, true));
        Assert.Equal(7808 * Mb, LongCatRecipeSupport.MeasuredReservedPeak(DeviceType.Cuda, 640, 368, 20, 4, true));
    }

    [Fact]
    public void TheMeasuredKeyIsThePlanThatWasMeasured()
    {
        // The key's latents are the plan's own: a 24-latent segment with 4 context latents has 20 new latents.
        LongCatAvatarChainPlan plan = LongCatAvatarChainPlan.For(24, 241);
        Assert.Equal(20, plan.NewLatentsPerSegment);
        Assert.Equal(4, plan.ContextLatents);
        Assert.Equal(11360 * Mb, Peak(832, 480, plan.NewLatentsPerSegment, plan.ContextLatents, distilled: true));
    }

    [Fact]
    public void A832x480LongerSegmentFallsBackToTheEstimateAndItsRefusal()
    {
        // A segment of 24 new latents (more than the measured 20) has no measurement: its tokens and the estimate apply, and at 14639 MiB it is refused.
        Assert.Null(Peak(832, 480, newLatents: 24));
        long tokens = LongCatRecipeSupport.TokenCount(24, 480 / 8, 832 / 8);
        long cache = Cached(832, 480);
        long headroom = LongCatRecipeSupport.BudgetHeadroomBytes(tokens, 4096, 11008, Peak(832, 480, newLatents: 24), cache);
        Assert.Equal(LongCatRecipeSupport.ActivationHeadroomBytes(tokens, 4096, 11008) + cache, headroom);
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            LongCatRecipeSupport.RequireRoom(14_639 * Mb, sharedBytesToPlace: 0, headroomBytes: headroom - cache, cacheBytes: cache, "The later run"));
        Assert.Contains("MiB short", error.Message);
    }

    [Fact]
    public void A832x480LongerContextFallsBackToTheEstimate()
    {
        Assert.Null(Peak(832, 480, contextLatents: 8));
        Assert.Null(Peak(640, 368, newLatents: 24));
    }

    [Fact]
    public void TheMarginIsTheSafetyMarginAndTheCachedStepIsOnTopOfIt()
    {
        Assert.Equal(LongCatRecipeSupport.SafetyBytes, LongCatRecipeSupport.MeasuredMarginBytes);
        Assert.Equal(7808 * Mb + LongCatRecipeSupport.SafetyBytes + 288 * Mb,
            LongCatRecipeSupport.BudgetHeadroomBytes(NoisyTokens(640, 368), 4096, 11008, 7808 * Mb, 288 * Mb));
    }

    [Fact]
    public void A640x368LaterRunHasSpareAndSevenBecomeEighteenResidentBlocks()
    {
        long cache = Cached(640, 368);
        long headroom = Measured(640, 368, cache);
        long free = 14_982 * Mb;
        long spare = free - headroom;

        // 7808 MiB reserved + 1024 MiB margin + the cached step: about 5860 MiB of spare, so the run is accepted with room to spare.
        Assert.InRange(headroom / (double)Mb, 9_100.0, 9_140.0);
        Assert.InRange(spare / (double)Mb, 5_840.0, 5_880.0);
        Assert.Equal(18, ResidentPrefixSizing.Size(free, headroom, Blocks));
    }

    [Fact]
    public void A832x480LaterRunHasSpareAtBothFreeFiguresAndFiveResidentBlocks()
    {
        long cache = Cached(832, 480);
        long headroom = Measured(832, 480, cache);
        // 14780 MiB is the measured run's planner free; 14639 MiB is the figure the estimate's refusal logged for the same size.
        foreach (long freeMb in new long[] { 14_780, 14_639 })
        {
            long free = freeMb * Mb;
            Assert.True(free - headroom >= 0, $"{freeMb} MiB free should cover the measured budget");
            Assert.Equal(5, ResidentPrefixSizing.Size(free, headroom, Blocks));
        }
        Assert.InRange((14_639 * Mb - headroom) / (double)Mb, 1_750.0, 1_790.0);
    }

    [Fact]
    public void TheMeasuredBudgetIsSmallerThanTheEstimateAtBothSizes()
    {
        Assert.True(Measured(640, 368, Cached(640, 368)) < Estimate(640, 368, Cached(640, 368)));
        Assert.True(Measured(832, 480, Cached(832, 480)) < Estimate(832, 480, Cached(832, 480)));
    }

    [Fact]
    public void AnUnmeasuredSizeKeepsTheEstimateAndItsRefusal()
    {
        long cache = Cached(1024, 576);
        Assert.Null(Peak(1024, 576));
        long headroom = Estimate(1024, 576, cache);
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            LongCatRecipeSupport.RequireRoom(14_639 * Mb, sharedBytesToPlace: 0, headroomBytes: headroom - cache, cacheBytes: cache, "The later run"));
        Assert.Contains("MiB short", error.Message);
    }

    [Fact]
    public void TheBeforeSegmentZeroBoundUsesTheMeasuredPeakThroughProduction()
    {
        LongCatConfig config = Config();
        using LongCatTransformer dit = new(config);
        LongCatParts parts = new(dit, 4, 48, false, null!, null!, null!, null!);
        long tokens = NoisyTokens(832, 480);
        long cache = Cached(832, 480);
        LongCatRoomCheckTests.OrderRecorder backend = LongCatRoomCheckTests.OrderRecorder.Create();
        long shared = LongCatRecipeSupport.NonResidentSharedBytes(backend.Backend, dit);

        // Before segment 0 the free memory is higher than the later run's measured planner free: this bound is an upper bound, and it passes.
        backend.FreeBytes = shared + 16_640 * Mb;
        LongCatRecipeSupport.RequireChainRoom(backend.Backend, parts, tokens, cache, 11_360 * Mb);

        // Well short of the measured budget: refused, and the message names the cache apart from the headroom.
        backend.FreeBytes = shared + 12_000 * Mb;
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            LongCatRecipeSupport.RequireChainRoom(backend.Backend, parts, tokens, cache, 11_360 * Mb));
        Assert.Contains("MiB of KV cache", error.Message);
    }
}
