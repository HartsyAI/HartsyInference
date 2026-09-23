using HartsyInference.Core.MemoryManagement;
using HartsyInference.Core.Models;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning.Memory;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Header-only VRAM estimation: inventory sizing, the Wan recipe hook, the per-tier fit verdicts, and the
/// once-per-checkpoint profile cache. Deterministic and GPU-free — every device fact is passed in.</summary>
public sealed class MemoryEstimationTests : IDisposable
{
    private const long Gib = 1L << 30;

    private readonly string _tempDir;

    public MemoryEstimationTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"hartsy-memory-estimate-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        try
        {
            Directory.Delete(_tempDir, recursive: true);
        }
        catch (IOException)
        {
        }
    }

    [Fact]
    public void Inventory_SeparatesBundledComponentsByPrefix()
    {
        CheckpointWeightInventory inventory = CheckpointWeightInventory.FromDescriptors(
        [
            Descriptor("model.diffusion_model.blocks.0.attn.weight", DType.F16, 64, 64),
            Descriptor("first_stage_model.decoder.conv.weight", DType.F32, 8, 8),
            Descriptor("cond_stage_model.transformer.weight", DType.F16, 16, 16),
        ]);

        Assert.Equal(64L * 64 * 2, inventory.StoredBytes(MemoryComponent.Denoiser));
        Assert.Equal(8L * 8 * 4, inventory.StoredBytes(MemoryComponent.Vae));
        Assert.Equal(16L * 16 * 2, inventory.StoredBytes(MemoryComponent.TextEncoder));
    }

    [Fact]
    public void Inventory_WidensQuantizedWeightsTheDeviceCannotHoldPacked()
    {
        CheckpointWeightInventory inventory = CheckpointWeightInventory.FromDescriptors(
        [
            Descriptor("blocks.0.ffn.weight", DType.Q4_K, 256, 256),
            Descriptor("blocks.0.conv.weight", DType.Q4_K, 256, 1, 1, 1),
        ]);
        long packedMatrix = DType.Q4_K.ComputeByteCount(256 * 256);

        long onPackedDevice = inventory.ResidentBytes(MemoryComponent.Denoiser, dtype => dtype == DType.Q4_K);
        long onPlainDevice = inventory.ResidentBytes(MemoryComponent.Denoiser, _ => false);

        // A packed-capable device keeps the matrix packed but still widens the non-matrix tensor no GEMM reads.
        Assert.Equal(packedMatrix + 256L * 2, onPackedDevice);
        Assert.Equal((256L * 256 + 256) * 2, onPlainDevice);
    }

    [Fact]
    public void Inventory_StreamFloorIsSharedWeightsPlusTheWindow()
    {
        List<SafeTensorDescriptor> descriptors = [Descriptor("patch_embedding.weight", DType.F16, 1024, 16)];
        for (int i = 0; i < 10; i++)
        {
            descriptors.Add(Descriptor($"blocks.{i}.ffn.weight", DType.F16, 1024, 1024));
        }
        CheckpointWeightInventory inventory = CheckpointWeightInventory.FromDescriptors(descriptors);
        long block = 1024L * 1024 * 2;
        long shared = 1024L * 16 * 2;

        Assert.Equal(10, inventory.DenoiserBlockCount);
        Assert.Equal(shared + 3 * block, inventory.DenoiserStreamFloorBytes(_ => true, windowBlocks: 3));
    }

    [Fact]
    public void WanRecipe_DescribesMemoryWithThePipelineFormulas()
    {
        const int inner = 5120;
        CheckpointHeader header = Header(
            Descriptor("patch_embedding.weight", DType.BF16, inner, 16, 1, 2, 2),
            Descriptor("head.head.weight", DType.BF16, 16 * 4, inner));

        RecipeMemoryModel? model = new WanVideoRecipe(WanVideoRecipe.Wan21_14BCompatClassId).DescribeMemory(header);

        Assert.NotNull(model);
        MemoryEstimateRequest request = new(Width: 1280, Height: 720, Frames: 81);
        long tokens = 21L * (720 / 8) * (1280 / 8);   // Wan2.1: 4x temporal, 8x spatial
        Assert.Equal(tokens * inner * 4 * 8 + 1536L * 1024 * 1024, model!.DenoiserActivationBytes(request));
        Assert.Equal(Math.Max(3 * Gib, 81L * 720 * 1280 * 160), model.VaeActivationBytes!(request));
        Assert.Equal(SideModels.Wan21Vae, model.Vae);
    }

    [Fact]
    public void WanRecipe_ReadsTheModelHeadNotAVaceBlockProjection()
    {
        const int inner = 5120;
        CheckpointHeader header = Header(
            Descriptor("model.diffusion_model.vace_blocks.0.proj_out.weight", DType.BF16, inner, inner),
            Descriptor("model.diffusion_model.patch_embedding.weight", DType.BF16, inner, 16, 1, 2, 2),
            Descriptor("model.diffusion_model.head.head.weight", DType.BF16, 16 * 4, inner));

        RecipeMemoryModel? model = new WanVideoRecipe(WanVideoRecipe.Wan21_14BCompatClassId).DescribeMemory(header);

        Assert.Equal(SideModels.Wan21Vae, model!.Vae);
    }

    [Fact]
    public void WanRecipe_UnreadableHeaderFallsBackToNoModel()
    {
        CheckpointHeader header = Header(Descriptor("something.else.weight", DType.F16, 4, 4));

        Assert.Null(new WanVideoRecipe().DescribeMemory(header));
    }

    [Fact]
    public void Estimate_FoldsPhasesByUnloadAndPlacement()
    {
        MemoryEstimate estimate = Estimate(textEncoder: 6 * Gib, denoiser: 10 * Gib, activation: 2 * Gib, vae: Gib);

        Assert.Equal(12 * Gib, estimate.PeakBytes(unloadBetweenPhases: true, _ => true));
        // Resident together: every weight plus the largest single working set.
        Assert.Equal(17 * Gib + 2 * Gib, estimate.PeakBytes(unloadBetweenPhases: false, _ => true));
        Assert.Equal(11 * Gib + 2 * Gib, estimate.PeakBytes(false, component => component != MemoryComponent.TextEncoder));
    }

    [Theory]
    [InlineData(24, MemoryFitVerdict.Resident)]
    [InlineData(10, MemoryFitVerdict.Streamed)]
    [InlineData(3, MemoryFitVerdict.Infeasible)]
    public void Judge_AutoTierPrefersResidentThenStreams(long usableGib, MemoryFitVerdict expected)
    {
        MemoryFit fit = MemoryFitJudge.Judge(Estimate(0, denoiser: 16 * Gib, activation: 2 * Gib, vae: Gib,
            streamFloor: 3 * Gib), VramPolicy.For(VramTier.Auto), canStream: true, usableGib * Gib, usableGib * Gib, _ => true);

        Assert.Equal(expected, fit.Verdict);
        Assert.Equal(VramTier.Auto, fit.EffectiveTier);
    }

    [Fact]
    public void Judge_PerformanceNeverStreamsAndHoldsEveryPhase()
    {
        MemoryEstimate estimate = Estimate(textEncoder: 6 * Gib, denoiser: 16 * Gib, activation: 2 * Gib, vae: Gib,
            streamFloor: 3 * Gib);

        MemoryFit fit = MemoryFitJudge.Judge(estimate, VramPolicy.For(VramTier.Performance), canStream: true,
            20 * Gib, 20 * Gib, _ => true);

        // 18 GB would fit phase by phase, but Performance keeps all 23 GB of weights resident and never streams.
        Assert.Equal(MemoryFitVerdict.Infeasible, fit.Verdict);
    }

    [Fact]
    public void Judge_RequestTierOverrideWinsOverTheBackendPolicy()
    {
        MemoryEstimate estimate = Estimate(0, denoiser: 16 * Gib, activation: 2 * Gib, vae: Gib, streamFloor: 3 * Gib);
        VramPolicy overridden = VramPolicyResolver.Apply(VramPolicy.For(VramTier.Aggressive),
            new VramOverrides { Tier = VramTier.Performance });

        MemoryFit fit = MemoryFitJudge.Judge(estimate, overridden, canStream: true, 10 * Gib, 10 * Gib, _ => true);

        Assert.Equal(VramTier.Performance, fit.EffectiveTier);
        Assert.Equal(MemoryFitVerdict.Infeasible, fit.Verdict);
    }

    [Fact]
    public void Judge_ModelWithoutStreamingCannotStreamWhateverThePolicy()
    {
        MemoryEstimate estimate = Estimate(0, denoiser: 16 * Gib, activation: 2 * Gib, vae: Gib, streamFloor: 3 * Gib);

        MemoryFit fit = MemoryFitJudge.Judge(estimate, VramPolicy.For(VramTier.Aggressive), canStream: false,
            10 * Gib, 10 * Gib, _ => true);

        Assert.Equal(MemoryFitVerdict.Infeasible, fit.Verdict);
    }

    [Fact]
    public void Judge_ShardPoolingOnlyCountsForTheDenoiser()
    {
        MemoryEstimate estimate = Estimate(0, denoiser: 30 * Gib, activation: 2 * Gib, vae: 20 * Gib);

        MemoryFit pooled = MemoryFitJudge.Judge(estimate, VramPolicy.For(VramTier.Auto), canStream: false,
            primaryBytes: 22 * Gib, denoiserBytes: 44 * Gib, _ => true);
        MemoryFit vaeTooBig = MemoryFitJudge.Judge(estimate with { Phases = [.. estimate.Phases.Select(p =>
            p.Component == MemoryComponent.Vae ? p with { WeightBytes = 23 * Gib } : p)] },
            VramPolicy.For(VramTier.Auto), canStream: false, primaryBytes: 22 * Gib, denoiserBytes: 44 * Gib, _ => true);

        Assert.Equal(MemoryFitVerdict.Resident, pooled.Verdict);
        Assert.Equal(MemoryFitVerdict.Infeasible, vaeTooBig.Verdict);
    }

    [Fact]
    public void Profile_ReadsEachCheckpointOnceAndRereadsWhenItChanges()
    {
        string path = WriteWanCheckpoint("wan.safetensors", inner: 256);
        ModelSpec spec = new()
        {
            Requested = WanVideoRecipe.Wan21_14BCompatClassId,
            Modality = Modality.Video,
            LocalPath = path,
        };

        CheckpointMemoryProfile[] concurrent = new CheckpointMemoryProfile[16];
        Parallel.For(0, concurrent.Length, i => concurrent[i] = CheckpointMemoryProfile.For(spec));
        Assert.All(concurrent, profile => Assert.Same(concurrent[0], profile));

        WriteWanCheckpoint("wan.safetensors", inner: 512);
        File.SetLastWriteTimeUtc(path, DateTime.UtcNow.AddMinutes(1));
        CheckpointMemoryProfile reread = CheckpointMemoryProfile.For(spec);

        Assert.NotSame(concurrent[0], reread);
        MemoryEstimate estimate = reread.Estimate(new MemoryEstimateRequest(832, 480, 33), _ => true);
        Assert.Equal(MemoryEstimateAccuracy.Recipe, estimate.Accuracy);
        Assert.Contains(estimate.Phases, phase => phase.Component == MemoryComponent.Denoiser && phase.Streamable);
    }

    [Fact]
    public async Task Service_DeviceWithoutMemoryReportIsUnknown()
    {
        using InferenceEngine engine = new("cpu");
        ModelSpec spec = new() { Requested = "flux1", Modality = Modality.Image, LocalPath = "unused.safetensors" };

        MemoryFit fit = await engine.MemoryEstimation.AssessAsync(spec, new MemoryEstimateRequest(1024, 1024));

        Assert.Equal(MemoryFitVerdict.Unknown, fit.Verdict);
    }

    private string WriteWanCheckpoint(string name, int inner)
    {
        string path = Path.Combine(_tempDir, name);
        using Tensor patch = new(new TensorShape([inner, 16, 1, 2, 2]), DType.F16);
        using Tensor head = new(new TensorShape(16 * 4, inner), DType.F16);
        using Tensor block = new(new TensorShape(inner, inner), DType.F16);
        SafeTensorsWriter.Save(path, new Dictionary<string, Tensor>
        {
            ["patch_embedding.weight"] = patch,
            ["head.head.weight"] = head,
            ["blocks.0.ffn.0.weight"] = block,
        });
        return path;
    }

    private static MemoryEstimate Estimate(long textEncoder, long denoiser, long activation, long vae, long streamFloor = 0)
    {
        List<MemoryPhase> phases = [];
        if (textEncoder > 0)
        {
            phases.Add(new MemoryPhase(MemoryComponent.TextEncoder, textEncoder, 0, 0, false));
        }
        phases.Add(new MemoryPhase(MemoryComponent.Denoiser, denoiser, activation, streamFloor, streamFloor > 0));
        phases.Add(new MemoryPhase(MemoryComponent.Vae, vae, Gib, 0, false));
        return new MemoryEstimate { FamilyId = "test", Phases = phases, Accuracy = MemoryEstimateAccuracy.Recipe };
    }

    private static CheckpointHeader Header(params SafeTensorDescriptor[] descriptors) =>
        new(ModelFormat.SafeTensors, descriptors.ToDictionary(d => d.Name), new Dictionary<string, string>());

    private static SafeTensorDescriptor Descriptor(string name, DType dtype, params long[] shape)
    {
        TensorShape tensorShape = new(shape);
        return new SafeTensorDescriptor
        {
            Name = name,
            DType = dtype,
            Shape = tensorShape,
            DataOffset = 0,
            ByteLength = dtype.ComputeByteCount(tensorShape.ElementCount),
        };
    }
}
