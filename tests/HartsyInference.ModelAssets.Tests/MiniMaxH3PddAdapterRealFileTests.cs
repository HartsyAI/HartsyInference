using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.MiniMaxH3;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Loads the real, hash-verified official PDD adapter (alibaba-pai/MiniMax-H3-Acc-LoRAs,
/// MiniMax-H3-FL2VA-Acc-8Step.safetensors) end to end through <see cref="MiniMaxH3PddAdapter.Load"/>. CPU-only and
/// independent of the multi-gigabyte DiT base checkpoint — this loader only reads the ~1.3 GB adapter file, so it
/// exercises the BF16-bank promotion and hash-bound task binding fixes without needing CUDA or a VRAM-fitting base.
/// The real file ships its projection banks as BF16 and carries no pdd_task/pdd_partition/hartsy.pdd.task metadata
/// key at all, which is exactly what <see cref="MiniMaxH3PddAdapter.Load"/> previously could not handle.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class MiniMaxH3PddAdapterRealFileTests
{
    private readonly ITestOutputHelper _output;
    public MiniMaxH3PddAdapterRealFileTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void RealFl2VaAdapter_LoadsWithBf16BanksPromotedToF32AndHashBoundTask()
    {
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.MiniMaxH3.PddFl2VaAdapter)) return;

        using MiniMaxH3PddAdapter adapter = MiniMaxH3PddAdapter.Load(
            TestPaths.MiniMaxH3.PddFl2VaAdapter, hashBoundTask: MiniMaxH3PddTask.Fl2Va);

        // The bank tensors ship BF16 on disk; the adapter must have promoted them to F32 exactly once.
        Assert.Equal(DType.F32, adapter.VideoHeadWeight.DType);
        Assert.Equal(DType.F32, adapter.VideoHeadBias.DType);
        Assert.Equal(DType.F32, adapter.AudioHeadWeight.DType);
        Assert.Equal(DType.F32, adapter.AudioHeadBias.DType);

        Assert.Equal(32, adapter.HeadBank.StepCount);
        Assert.Equal(PddHeadBank.PublishedVideoChannels, adapter.HeadBank.VideoChannels);
        Assert.Equal(PddHeadBank.PublishedAudioChannels, adapter.HeadBank.AudioChannels);
        Assert.Equal(PddHeadBank.PublishedHiddenSize, adapter.HeadBank.HiddenSize);
        Assert.Equal(32, adapter.PddNumSteps);
        Assert.Equal(4, adapter.PddBlockSize);
        Assert.Equal(64, adapter.Rank);

        // The real file has no pdd_task/pdd_partition/hartsy.pdd.task key — the hash-bound task passed by the
        // caller (mirroring VideoProfileResolver's manifest-hash lookup) must be what actually lands.
        Assert.Equal(MiniMaxH3PddTask.Fl2Va, adapter.Task);
        Assert.DoesNotContain("pdd_task", adapter.Metadata!.Keys);
        Assert.DoesNotContain("pdd_partition", adapter.Metadata!.Keys);

        // Every one of the 258 non-head trunk targets converted with no tensor silently skipped.
        Assert.Equal(258, adapter.Trunk.Layers.Count);

        // One finite element from each promoted bank row confirms the cast actually copied real data, not zeros.
        Assert.True(float.IsFinite(adapter.HeadBank.GetVideoWeight(0).AsSpan<float>()[0]));
        Assert.True(float.IsFinite(adapter.HeadBank.GetAudioWeight(31).AsSpan<float>()[0]));

        _output.WriteLine($"Loaded: {adapter.HeadBank.StepCount} steps, {adapter.Trunk.Layers.Count} trunk "
            + $"targets, task={adapter.Task}, rank={adapter.Rank}, alpha={adapter.Alpha}.");
    }
}
