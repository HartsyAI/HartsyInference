using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The chain's budget refusals. The later run's refusal is the scope's own check (LongCatRecipeSupport.OpenDenoiseStream, which calls
/// RequireRoom with the staged pair counted in its headroom). Before segment 0, RequireChainRoom is an upper bound that refuses only when even that
/// bound is negative. Figures are MiB: a 832x480 later run needs 15894 MiB of activation headroom without the staged pair, and about 16138 with it.</summary>
public sealed class LongCatRoomCheckTests
{
    private const long Mb = 1L << 20;

    private static long LaterTokens(int width, int height)
    {
        LongCatAvatarChainPlan plan = LongCatAvatarChainPlan.For(24, 241);
        return LongCatRecipeSupport.TokenCount(plan.RunLatents, height / 8, width / 8);
    }

    private static LongCatConfig Config() => new() { HiddenSize = 4096, Depth = 48, PatchSize = (1, 2, 2) };

    [Fact]
    public void ALaterRunAt832x480WithoutThePairIsRefusedWithItsFigures()
    {
        long headroom = LongCatRecipeSupport.ActivationHeadroomBytes(LaterTokens(832, 480), 4096, 11008);
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            LongCatRecipeSupport.RequireRoom(14_639 * Mb, sharedBytesToPlace: 0, headroomBytes: headroom, "The later run"));
        Assert.Contains("MiB short", error.Message);
        Assert.Contains("14639 MiB is free", error.Message);
    }

    [Fact]
    public void ALaterRunAt640x368FitsWithItsFreeMemory()
    {
        long headroom = LongCatRecipeSupport.ActivationHeadroomBytes(LaterTokens(640, 368), 4096, 11008);
        LongCatRecipeSupport.RequireRoom(14_550 * Mb, sharedBytesToPlace: 0, headroomBytes: headroom, "The later run");
    }

    [Fact]
    public void TheOffloadedStagedPairCountsAgainstThe832x480LaterRun()
    {
        LongCatConfig config = Config();
        long stage = LongCatRecipeSupport.KvCacheBytes(config, 5, 60, 104, offloaded: true);
        long headroom = LongCatRecipeSupport.ActivationHeadroomBytes(LaterTokens(832, 480), 4096, 11008) + stage;

        // 14680 MiB free: refused by about 1458 MiB once the pair is counted.
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            LongCatRecipeSupport.RequireRoom(14_680 * Mb, sharedBytesToPlace: 0, headroomBytes: headroom, "The later run"));
        Assert.Contains("MiB short", error.Message);
    }

    [Fact]
    public void A832x480RunThatFitsWithoutThePairIsRefusedWithIt()
    {
        // 15900 MiB free sits between the two figures: about 15894 MiB without the staged pair (6 MiB spare), about 16138 MiB with it (short).
        // A check that did not count the pair would accept this run, so this test fails if the pair is dropped.
        LongCatConfig config = Config();
        long without = LongCatRecipeSupport.ActivationHeadroomBytes(LaterTokens(832, 480), 4096, 11008);
        long with = without + LongCatRecipeSupport.KvCacheBytes(config, 5, 60, 104, offloaded: true);

        LongCatRecipeSupport.RequireRoom(15_900 * Mb, sharedBytesToPlace: 0, headroomBytes: without, "The later run");
        Assert.Throws<InvalidOperationException>(() =>
            LongCatRecipeSupport.RequireRoom(15_900 * Mb, sharedBytesToPlace: 0, headroomBytes: with, "The later run"));
    }

    [Fact]
    public void TheOffloadedStagedPairStillFitsA640x368LaterRun()
    {
        LongCatConfig config = Config();
        long stage = LongCatRecipeSupport.KvCacheBytes(config, 5, 46, 80, offloaded: true);
        long headroom = LongCatRecipeSupport.ActivationHeadroomBytes(LaterTokens(640, 368), 4096, 11008) + stage;
        LongCatRecipeSupport.RequireRoom(14_761 * Mb, sharedBytesToPlace: 0, headroomBytes: headroom, "The later run");
    }

    [Fact]
    public void TheSharedWeightsStillToPlaceCountAgainstTheRoom()
    {
        long headroom = LongCatRecipeSupport.ActivationHeadroomBytes(LaterTokens(640, 368), 4096, 11008);
        // 14550 MiB free fits on its own; 3000 MiB of shared weights still to place does not.
        Assert.Throws<InvalidOperationException>(() =>
            LongCatRecipeSupport.RequireRoom(14_550 * Mb, sharedBytesToPlace: 3_000 * Mb, headroomBytes: headroom, "The later run"));
    }

    [Fact]
    public void ZeroSpendableIsAccepted()
    {
        LongCatRecipeSupport.RequireRoom(10 * Mb, sharedBytesToPlace: 4 * Mb, headroomBytes: 6 * Mb, "The later run");
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
        Assert.Equal(["TrimMemoryPool", "FreeMemoryBytes"], backend.Calls.Take(2).ToArray());

        // A roomy card passes.
        backend.FreeBytes = 40L << 30;
        LongCatRecipeSupport.RequireChainRoom(backend.Backend, parts, 1_000, 0);
    }

    private static OrderRecorder Recorder() => OrderRecorder.Create();

    /// <summary>A backend that answers the memory calls the check makes and records their order; everything else goes to the CPU backend.</summary>
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
                default:
                    return targetMethod.Invoke(_cpu, args);
            }
        }
    }
}
