using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Recipes.Video;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>The chain's budget refusal: a later run whose free memory, less its shared weights and its activation headroom, is negative is refused
/// before its denoise, and the same check runs before segment 0 as an upper bound. The figures are the 4090 logs: a 832x480 later run had 14639 MiB
/// free and needs more than that; a 640x368 later run had 14550 MiB free and fits.</summary>
public sealed class LongCatRoomCheckTests
{
    private const long Mb = 1L << 20;

    [Fact]
    public void ALaterRunAt832x480IsRefusedWithItsFigures()
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
    public void TheSharedWeightsStillToPlaceCountAgainstTheRoom()
    {
        long headroom = LongCatRecipeSupport.ActivationHeadroomBytes(LaterTokens(640, 368), 4096, 11008);
        // 14550 MiB free fits on its own; 3000 MiB of shared weights still to place does not.
        Assert.Throws<InvalidOperationException>(() =>
            LongCatRecipeSupport.RequireRoom(14_550 * Mb, sharedBytesToPlace: 3_000 * Mb, headroomBytes: headroom, "The later run"));
    }

    [Fact]
    public void TheLaterRunPlacesItsSharedWeightsThenTrimsThenReads()
    {
        using LongCatTransformer dit = new(new LongCatConfig());
        LongCatParts parts = new(dit, 4, 48, false, null!, null!, null!, null!);
        OrderRecorder backend = Recorder();
        backend.FreeBytes = 40L << 30;

        LongCatRecipeSupport.RequireLaterRunRoom(backend.Backend, parts, LaterTokens(640, 368), 0);

        Assert.Equal(["PreloadWeights", "TrimMemoryPool", "FreeMemoryBytes"], backend.Calls.ToArray());
    }

    [Fact]
    public void TheOffloadedStagedPairCountsAgainstThe832x480LaterRun()
    {
        LongCatConfig config = new() { HiddenSize = 4096, Depth = 48, PatchSize = (1, 2, 2) };
        long stage = LongCatRecipeSupport.KvCacheBytes(config, 5, 60, 104, offloaded: true);
        long headroom = LongCatRecipeSupport.ActivationHeadroomBytes(LaterTokens(832, 480), 4096, 11008) + stage;
        // 14680 MiB free: the earlier check refused with 1215 MiB short, and the staged pair adds about 244 MiB more.
        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            LongCatRecipeSupport.RequireRoom(14_680 * Mb, sharedBytesToPlace: 0, headroomBytes: headroom, "The later run"));
        Assert.Contains("MiB short", error.Message);
    }

    [Fact]
    public void TheOffloadedStagedPairStillFitsA640x368LaterRun()
    {
        LongCatConfig config = new() { HiddenSize = 4096, Depth = 48, PatchSize = (1, 2, 2) };
        long stage = LongCatRecipeSupport.KvCacheBytes(config, 5, 46, 80, offloaded: true);
        long headroom = LongCatRecipeSupport.ActivationHeadroomBytes(LaterTokens(640, 368), 4096, 11008) + stage;
        LongCatRecipeSupport.RequireRoom(14_761 * Mb, sharedBytesToPlace: 0, headroomBytes: headroom, "The later run");
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

    /// <summary>The later run's token rows at a size, from a real chain plan: its run latents times the latent grid's tokens per frame.</summary>
    private static long LaterTokens(int width, int height)
    {
        LongCatAvatarChainPlan plan = LongCatAvatarChainPlan.For(24, 241);
        return LongCatRecipeSupport.TokenCount(plan.RunLatents, height / 8, width / 8);
    }

    private static OrderRecorder Recorder() => OrderRecorder.Create();

    /// <summary>A backend that answers the two memory calls the check makes and records their order; everything else goes to the CPU backend.</summary>
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
                case nameof(IBackend.PreloadWeights):
                    Calls.Add(nameof(IBackend.PreloadWeights));
                    return targetMethod.Invoke(_cpu, args);
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
