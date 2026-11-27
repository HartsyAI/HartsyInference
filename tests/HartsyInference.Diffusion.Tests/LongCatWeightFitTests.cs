using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Memory.Vram;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine;
using HartsyInference.Engine.Placement;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.Tests.Shared;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A backend that streams no weights is refused a LongCat DiT it cannot hold: by the planner through the video service and again by the
/// recipe at load, on tiny releases with a stand-in backend that reports a device size (everything else is the CPU's).</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatWeightFitTests : IDisposable
{
    private readonly string _base = Path.Combine(Path.GetTempPath(), "longcat-fit-" + Guid.NewGuid().ToString("N"), "dit");
    private readonly LongCatTinyAvatarRelease _int8 = new(onePointSix: true);

    public LongCatWeightFitTests()
    {
        Directory.CreateDirectory(_base);
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(LongCatTestData.Tiny, seed: 5);
        LongCatTestData.WriteCheckpoint(_base, weights, storage: DType.BF16);
        LongCatTestData.DisposeAll(weights);
    }

    public void Dispose()
    {
        _int8.Dispose();
        Directory.Delete(Path.GetDirectoryName(_base)!, recursive: true);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void TheHeaderFigureIsTheBytesOfTheWeightsAsTheyAreHeld(bool int8, bool packed)
    {
        string directory = int8 ? _int8.Checkpoint : _base;
        using LongCatCheckpoint checkpoint = LongCatCheckpoint.Open(directory);
        Dictionary<string, Tensor> weights = new(checkpoint.Weights, StringComparer.Ordinal);
        // The widening LoadParts runs, with a backend that holds every packed weight or none.
        using QuantizedWeightPolicy.PreparedWeights prepared = QuantizedWeightPolicy.PrepareFor(weights, _ => packed, "stand-in", DType.F32);

        long held = weights.Values.Sum(static t => t.DType.ComputeByteCount(t.ElementCount)
            + (t.QuantInfo?.RowScale is { } scales ? scales.DType.ComputeByteCount(scales.ElementCount) : 0));
        (long bytes, long widened) = LongCatWeightFit.HeldBytes(directory, _ => packed);

        Assert.Equal(held, bytes);
        Assert.Equal(prepared.WidenedCount, widened);
        Assert.Equal(int8 && !packed, widened > 0);
    }

    [Theory]
    [InlineData("longcat-video", false)]
    [InlineData("longcat-avatar-1.6", true)]
    public async Task APlanOnABackendThatCannotHoldTheWeightsIsRefusedWithBothSizes(string model, bool int8)
    {
        string checkpoint = int8 ? _int8.Checkpoint : _base;
        long held = LongCatWeightFit.HeldBytes(checkpoint, static _ => false).Bytes;
        // One byte short of the weights once the engine's per-device reserve is taken off.
        long total = PlacementPlanner.PerDeviceReserveBytes + held - 1;

        VideoPlan plan = await Plan(StandIn.Create(total, streams: false), model, checkpoint);

        VideoPlanIssue issue = Assert.Single(plan.Issues, static i => i.Severity == VideoPlanIssueSeverity.Error);
        Assert.Equal("video.memory.weights_do_not_fit", issue.Code);
        Assert.Contains($"the weights take {ByteFormat.GbF1(held)}", issue.Message, StringComparison.Ordinal);
        Assert.Contains($"{ByteFormat.GbF1(held - 1)} usable of {ByteFormat.GbF1(total)}", issue.Message, StringComparison.Ordinal);
        Assert.Equal(int8, issue.Message.Contains("widen to F32", StringComparison.Ordinal));
        Assert.False(plan.IsValid);
    }

    [Theory]
    [InlineData(0L, false)]
    [InlineData(-1L, true)]
    public async Task APlanPassesWhereTheWeightsFitOrStream(long spare, bool streams)
    {
        long held = LongCatWeightFit.HeldBytes(_base, static _ => false).Bytes;

        VideoPlan plan = await Plan(StandIn.Create(PlacementPlanner.PerDeviceReserveBytes + held + spare, streams), "longcat-video", _base);

        Assert.True(plan.IsValid, string.Join("; ", plan.Issues.Select(static i => i.Message)));
    }

    [Fact]
    public async Task APlanOnTheCpuIsNotRefused()
    {
        VideoPlan plan = await Plan(new CpuBackend(), "longcat-video", _base);

        Assert.True(plan.IsValid, string.Join("; ", plan.Issues.Select(static i => i.Message)));
    }

    [Fact]
    public void ABackendThatStreamsOrReportsNoMemoryIsAnsweredWithoutReadingTheCheckpoint()
    {
        string absent = Path.Combine(_base, "absent");
        using CpuBackend cpu = new();
        using IBackend streaming = StandIn.Create(1, streams: true);
        using IBackend resident = StandIn.Create(1L << 40, streams: false);

        Assert.Null(LongCatWeightFit.Refusal(cpu, absent));
        Assert.Null(LongCatWeightFit.Refusal(streaming, absent));
        Assert.Throws<FileNotFoundException>(() => LongCatWeightFit.Refusal(resident, absent));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ARecipeConstructedPastThePlannerIsRefusedBeforeItLoads(bool int8)
    {
        using IBackend backend = StandIn.Create(1, streams: false);
        IVideoRecipe recipe = int8 ? new LongCatAvatarRecipe() : new LongCatVideoRecipe();
        RecipeContext context = new()
        {
            CheckpointPath = int8 ? _int8.Checkpoint : _base, Backend = backend,
            VariantHints = int8 ? [LongCatAvatarVariants.Avatar16FamilyId] : [],
        };

        NotSupportedException refused = Assert.Throws<NotSupportedException>(() => recipe.Construct(context));

        Assert.Contains("no weight streaming", refused.Message, StringComparison.Ordinal);
    }

    [Trait("Category", "GpuIntegration")]
    [VulkanDeviceFact]
    public void ARealVulkanDeviceThatHoldsTheWeightsIsNotRefused()
    {
        using VulkanBackend vulkan = GpuDevices.OpenVulkan();

        Assert.Null(((IBackend)vulkan).StreamingCache);
        Assert.Null(LongCatWeightFit.Refusal(vulkan, _base));
        Assert.Null(LongCatWeightFit.Refusal(vulkan, _int8.Checkpoint));
    }

    /// <summary>Plans through the video service of an engine over <paramref name="backend"/>, which the engine disposes.</summary>
    private static async Task<VideoPlan> Plan(IBackend backend, string model, string checkpoint)
    {
        using InferenceEngine engine = new(backend, "cpu");
        return await engine.VideoPlanning.PlanAsync(ModelResolver.Resolve(model, checkpoint, Modality.Video),
            new VideoRequest { Prompt = "a lighthouse at dusk" });
    }

    /// <summary>A device backend as the check sees one: a total device memory and a streaming cache or none; everything else is the CPU's.</summary>
    public class StandIn : DispatchProxy
    {
        private readonly CpuBackend _cpu = new();
        private long _total;
        private bool _streams;

        public static IBackend Create(long totalBytes, bool streams)
        {
            IBackend proxy = DispatchProxy.Create<IBackend, StandIn>();
            StandIn standIn = (StandIn)(object)proxy;
            standIn._total = totalBytes;
            standIn._streams = streams;
            return proxy;
        }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            ArgumentNullException.ThrowIfNull(targetMethod);
            return targetMethod.Name switch
            {
                nameof(IBackend.GetVramInfo) => (_total, _total),
                // Only its presence is read.
                "get_StreamingCache" => _streams ? DispatchProxy.Create<IStreamingWeightCache, Unused>() : null,
                _ => targetMethod.Invoke(_cpu, args),
            };
        }
    }

    /// <summary>A streaming cache nothing calls.</summary>
    public class Unused : DispatchProxy
    {
        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args) =>
            throw new NotSupportedException($"{targetMethod?.Name} is not part of the check.");
    }
}
