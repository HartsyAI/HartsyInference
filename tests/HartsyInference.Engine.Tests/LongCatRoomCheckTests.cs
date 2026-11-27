using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The later run's budget is sized on its cached step: the new latents' tokens, the activations, and the cache's two pairs
/// (<see cref="LongCatRecipeSupport.CachedStepBytes"/>). The later run's own check (OpenDenoiseStream) and the pre-segment-0 bound
/// (RequireChainRoom) use the same figures. Figures are MiB: at 832x480 the activation headroom is about 14189 without the cached step and about
/// 14677 with it, so the 14639 MiB the log reports sits between the two.</summary>
public sealed class LongCatRoomCheckTests
{
    private const long Mb = 1L << 20;

    /// <summary>The later run's noisy tokens: the new latents of a segment, not the run's whole latent grid.</summary>
    private static long NoisyTokens(int width, int height)
    {
        LongCatAvatarChainPlan plan = LongCatAvatarChainPlan.For(24, 241);
        return LongCatRecipeSupport.TokenCount(plan.NewLatentsPerSegment, height / 8, width / 8);
    }

    /// <summary>The cached step's device bytes for five conditioning latents at the given size.</summary>
    private static long Cached(LongCatConfig config, int width, int height) =>
        LongCatRecipeSupport.CachedStepBytes(config, 5, height / 8, width / 8);

    private static long Activations(long tokens) => LongCatRecipeSupport.ActivationHeadroomBytes(tokens, 4096, 11008);

    private static LongCatConfig Config() => new() { HiddenSize = 4096, Depth = 48, PatchSize = (1, 2, 2) };

    [Fact]
    public void ALaterRunAt832x480IsRefusedOnlyWithItsCachedStep()
    {
        LongCatConfig config = Config();
        long tokens = NoisyTokens(832, 480);
        long without = Activations(tokens);
        long withCache = without + Cached(config, 832, 480);
        long free = 14_639 * Mb;

        // The free figure sits between the two headrooms: the activations fit alone, and the cached step does not.
        Assert.True(without < free && free < withCache, $"free {free / Mb} MiB should sit between {without / Mb} and {withCache / Mb} MiB");

        LongCatRecipeSupport.RequireRoom(free, sharedBytesToPlace: 0, headroomBytes: without, cacheBytes: 0, "The later run");
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            LongCatRecipeSupport.RequireRoom(free, sharedBytesToPlace: 0, headroomBytes: without, cacheBytes: Cached(config, 832, 480), "The later run"));
        Assert.Contains("MiB of KV cache", error.Message);
        Assert.Contains("14639 MiB is free", error.Message);
        Assert.Contains("MiB short", error.Message);
        Assert.DoesNotContain("shared weights", error.Message);
    }

    [Fact]
    public void ALaterRunAt640x368FitsWithItsCachedStep()
    {
        LongCatConfig config = Config();
        // 14761 MiB free, the #1230 segment-1 figure: about 3084 MiB spare once the cached step is counted.
        long cached = Cached(config, 640, 368);
        long spare = 14_761 * Mb - Activations(NoisyTokens(640, 368)) - cached;
        Assert.InRange(spare / (double)Mb, 3_070.0, 3_100.0);
        LongCatRecipeSupport.RequireRoom(14_761 * Mb, sharedBytesToPlace: 0, headroomBytes: Activations(NoisyTokens(640, 368)),
            cacheBytes: cached, "The later run");
    }

    [Fact]
    public void TheCachedStepCountsTwoPairsOfTheConditioningTokens()
    {
        LongCatConfig config = Config();
        long pair = LongCatRecipeSupport.KvCacheBytes(config, 5, 60, 104, offloaded: true);
        Assert.Equal(2 * pair, LongCatRecipeSupport.CachedStepBytes(config, 5, 60, 104));
    }

    [Fact]
    public void TheSharedWeightsStillToPlaceCountAgainstTheRoom()
    {
        long headroom = Activations(NoisyTokens(640, 368));
        // 14550 MiB free leaves about 3160 MiB past the activations, so 2000 MiB of shared weights still to place fits and 3500 MiB does not.
        LongCatRecipeSupport.RequireRoom(14_550 * Mb, sharedBytesToPlace: 2_000 * Mb, headroomBytes: headroom, cacheBytes: 0, "The later run");
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            LongCatRecipeSupport.RequireRoom(14_550 * Mb, sharedBytesToPlace: 3_500 * Mb, headroomBytes: headroom, cacheBytes: 0, "The later run"));
        Assert.Contains("3500 MiB of shared weights", error.Message);
    }

    [Fact]
    public void ZeroSpendableIsAccepted()
    {
        LongCatRecipeSupport.RequireRoom(10 * Mb, sharedBytesToPlace: 4 * Mb, headroomBytes: 6 * Mb, cacheBytes: 0, "The later run");
    }

    [Fact]
    public void TheBeforeSegmentZeroCheckCountsTheCachedStepThroughProduction()
    {
        // The production helper, through a recording backend, with the later run's own figures: noisy tokens and the cached step.
        LongCatConfig config = Config();
        using LongCatTransformer dit = new(config);
        LongCatParts parts = new(dit, 4, 48, false, null!, null!, null!, null!);
        long tokens = NoisyTokens(832, 480);
        long cached = Cached(config, 832, 480);
        long without = LongCatRecipeSupport.BudgetHeadroomBytes(tokens, config.HiddenSize, config.FfnHiddenDim, measuredPeakBytes: null, cacheBytes: 0);
        Assert.Equal(without + cached,
            LongCatRecipeSupport.BudgetHeadroomBytes(tokens, config.HiddenSize, config.FfnHiddenDim, measuredPeakBytes: null, cacheBytes: cached));

        OrderRecorder backend = Recorder();
        long shared = LongCatRecipeSupport.NonResidentSharedBytes(backend.Backend, dit);
        // Free memory sits between the two bounds: half the cached step to spare without it passes, and with the whole step it refuses.
        backend.FreeBytes = without + shared + cached / 2;

        LongCatRecipeSupport.RequireChainRoom(backend.Backend, parts, tokens, 0);
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            LongCatRecipeSupport.RequireChainRoom(backend.Backend, parts, tokens, cached));
        Assert.Contains("MiB of KV cache", error.Message);
        Assert.Contains("MiB short", error.Message);
        Assert.Equal(new[] { "TrimMemoryPool", "FreeMemoryBytes", "TrimMemoryPool", "FreeMemoryBytes" }, backend.Calls.ToArray());
    }

    [Fact]
    public void TheBeforeSegmentZeroCheckTrimsThePoolBeforeItReadsTheFreeMemory()
    {
        // An unloaded transformer has no shared weights to place, so the check is the fixed pool and the headroom alone.
        using LongCatTransformer dit = new(new LongCatConfig());
        LongCatParts parts = new(dit, 4, 48, false, null!, null!, null!, null!);
        OrderRecorder backend = Recorder();

        // Too little for even the fixed pool: refused, and the trim came first.
        Assert.Throws<InvalidOperationException>(() => LongCatRecipeSupport.RequireChainRoom(backend.Backend, parts, 1_000, 0));
        Assert.Equal(new[] { "TrimMemoryPool", "FreeMemoryBytes" }, backend.Calls.Take(2).ToArray());

        // A roomy card passes.
        backend.FreeBytes = 40L << 30;
        LongCatRecipeSupport.RequireChainRoom(backend.Backend, parts, 1_000, 0);
    }

    [Fact]
    public void ABackendWithoutDeviceMemoryIsNotRefused()
    {
        using LongCatTransformer dit = new(new LongCatConfig());
        LongCatParts parts = new(dit, 4, 48, false, null!, null!, null!, null!);
        using CpuBackend cpu = new();

        LongCatRecipeSupport.RequireChainRoom(cpu, parts, 1_000, 1L << 30);
    }

    private static OrderRecorder Recorder() => OrderRecorder.Create();

    /// <summary>A device backend as the check sees one: it answers the memory calls and records the order of the trim and the free read; everything else
    /// goes to the CPU backend.</summary>
    public class OrderRecorder : DispatchProxy
    {
        private readonly CpuBackend _cpu = new();

        public IBackend Backend { get; private set; } = null!;
        public List<string> Calls { get; } = [];
        public long FreeBytes { get; set; } = 1L << 20;

        public static OrderRecorder Create()
        {
            IBackend proxy = DispatchProxy.Create<IBackend, OrderRecorder>();
            OrderRecorder recorder = (OrderRecorder)(object)proxy;
            recorder.Backend = proxy;
            return recorder;
        }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            ArgumentNullException.ThrowIfNull(targetMethod);
            switch (targetMethod.Name)
            {
                case nameof(IBackend.TrimMemoryPool):
                    Calls.Add(nameof(IBackend.TrimMemoryPool));
                    return null;
                case nameof(IBackend.FreeMemoryBytes):
                    Calls.Add(nameof(IBackend.FreeMemoryBytes));
                    return FreeBytes;
                case nameof(IBackend.GetVramInfo):
                    return (FreeBytes, 1L << 45);
                default:
                    return targetMethod.Invoke(_cpu, args);
            }
        }
    }
}
