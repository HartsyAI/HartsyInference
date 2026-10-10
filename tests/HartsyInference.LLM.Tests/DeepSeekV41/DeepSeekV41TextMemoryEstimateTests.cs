using HartsyInference.Cpu;
using HartsyInference.Engine;
using HartsyInference.Engine.Services;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning.Memory;
using HartsyInference.Engine.Registry;
using HartsyInference.LLM.DeepSeekV41;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The header-only memory estimate of a text checkpoint: classes sum to the index total, and the phase is derived from them.</summary>
public sealed class DeepSeekV41TextMemoryEstimateTests : IDisposable
{
    private readonly string _directory = Directory.CreateTempSubdirectory("dsv41-estimate-").FullName;

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Fact]
    public void ClassBytesSumToTheIndexTotalSize()
    {
        long total = TinyDeepSeekV41Checkpoint.Write(_directory);

        MemoryEstimate estimate = TextMemoryProfile.Estimate(_directory);

        Assert.NotNull(estimate.WeightBytesByClass);
        Assert.Equal(Enum.GetValues<DeepSeekV41WeightClass>().Length, estimate.WeightBytesByClass!.Count);
        Assert.Equal(total, estimate.WeightBytesByClass.Values.Sum());
        Assert.Equal(MemoryEstimateAccuracy.HeaderOnly, estimate.Accuracy);
    }

    [Fact]
    public void ResidentPhaseCountsTheBackboneAndStreamFloorExcludesExpertsAndEngram()
    {
        TinyDeepSeekV41Checkpoint.Write(_directory);

        MemoryEstimate estimate = TextMemoryProfile.Estimate(_directory);
        IReadOnlyDictionary<string, long> bytes = estimate.WeightBytesByClass!;
        MemoryPhase phase = Assert.Single(estimate.Phases);

        Assert.Equal(MemoryComponent.LanguageModel, phase.Component);
        Assert.Equal(bytes["Dense"] + bytes["Expert"] + bytes["Engram"] + bytes["Embed"] + bytes["Head"], phase.WeightBytes);
        Assert.Equal(bytes["Dense"] + bytes["Embed"] + bytes["Head"], phase.StreamFloorWeightBytes);
        Assert.True(phase.Streamable);
        Assert.True(bytes["Vision"] > 0 && bytes["Draft"] > 0);
        Assert.Equal(phase.WeightBytes + phase.ActivationBytes, estimate.PeakResidentBytes);
    }

    [Fact]
    public void StaticSequenceStateBytesEqualTheLoadedModelsOwn()
    {
        DeepSeekV41ModelFixtureCheckpoint.Write(_directory);
        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory);
        using CpuBackend backend = new();
        using DeepSeekV41LoadedModel loaded = DeepSeekV41HostModelLoader.Load(backend, checkpoint, new DeepSeekV41LoadOptions(MaxTokens: 64));

        foreach (int tokens in new[] { 1, 17, 64 })
            Assert.Equal(loaded.Model.EstimateStateBytes(tokens), DeepSeekV41WorkingMemory.SequenceStateBytes(checkpoint.Config, tokens));
    }

    [Fact]
    public void ActivationsGrowWithThePromptAndWideningAddsFourTimesTheStoredDenseBytes()
    {
        DeepSeekV41ModelFixtureCheckpoint.Write(_directory);
        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory);
        DeepSeekV41Config cfg = checkpoint.Config;
        DeepSeekV41LoadOptions stored = new(MaxTokens: 128), widened = stored with { Residency = DeepSeekV41Residency.WidenedF32 };

        Assert.True(DeepSeekV41WorkingMemory.ActivationBytes(cfg, 200) > DeepSeekV41WorkingMemory.ActivationBytes(cfg, 100));
        Assert.Equal(4000L, DeepSeekV41WorkingMemory.AnonymousBytes(cfg, 1000, widened) - DeepSeekV41WorkingMemory.AnonymousBytes(cfg, 1000, stored)
            - DeepSeekV41WorkingMemory.ExpertCacheBytes(cfg, widened));
        Assert.Equal(0, DeepSeekV41WorkingMemory.ExpertCacheBytes(cfg, stored));
        Assert.True(DeepSeekV41WorkingMemory.SmallTensorBytes(cfg) > 0);
    }

    [Fact]
    public async Task Service_EstimateAndAssessRouteTextDirectoriesToTheTextProfile()
    {
        long total = TinyDeepSeekV41Checkpoint.Write(_directory);
        using InferenceEngine engine = new("cpu");
        ModelSpec spec = new() { Requested = DeepSeekV41Catalog.Id, Modality = Modality.Text, LocalPath = _directory };

        MemoryEstimate estimate = await engine.MemoryEstimation.EstimateAsync(spec, new MemoryEstimateRequest(1, 1));
        MemoryFit fit = await engine.MemoryEstimation.AssessAsync(spec, new MemoryEstimateRequest(1, 1));

        Assert.Equal(total, estimate.WeightBytesByClass!.Values.Sum());
        Assert.NotEqual(MemoryFitVerdict.Unknown, fit.Verdict);
        Assert.True(fit.CapacityBytes > 0);
        Assert.Contains("working memory", fit.Reason, StringComparison.Ordinal);
        Assert.NotNull(fit.Estimate);
    }

}
