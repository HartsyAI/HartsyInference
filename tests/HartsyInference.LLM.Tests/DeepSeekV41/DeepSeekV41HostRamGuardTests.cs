using HartsyInference.Core.Exceptions;
using HartsyInference.Engine.Placement;
using HartsyInference.Engine.Planning.Memory;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.DeepSeekV41;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The V4.1 host RAM guard refuses a load whose working set does not fit the free memory it is given, before any weight is mapped.</summary>
public sealed class DeepSeekV41HostRamGuardTests : IDisposable
{
    private const long KiBPerGiB = 1024L * 1024;

    private readonly string _directory = Directory.CreateTempSubdirectory("dsv41-ramguard-").FullName;

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Fact]
    public void FreeMemoryBelowTheWorkingSet_RefusesTheLoad()
    {
        TinyDeepSeekV41Checkpoint.Write(_directory);

        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(
            () => TextService.EnsureRamHeadroomForDeepSeekV41(_directory, availableKb: 1024));

        Assert.Contains("Not enough free host RAM", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void FreeMemoryAboveTheWorkingSet_Passes()
    {
        TinyDeepSeekV41Checkpoint.Write(_directory);

        TextService.EnsureRamHeadroomForDeepSeekV41(_directory, availableKb: 64 * KiBPerGiB);
    }

    [Fact]
    public void UnknownFreeMemory_DoesNotRefuse()
    {
        // The checkpoint is never opened when the free memory is unknown, so the directory need not exist.
        Assert.Null(TextService.EnsureRamHeadroomForDeepSeekV41(Path.Combine(_directory, "absent"), availableKb: 0));
    }

    [Fact]
    public void HostDemand_ChargesExactlyTheAnonymousBytesTheGuardRequires()
    {
        TinyDeepSeekV41Checkpoint.Write(_directory);
        using DeepSeekV41Checkpoint checkpoint = DeepSeekV41Checkpoint.Open(_directory);
        IReadOnlyDictionary<DeepSeekV41WeightClass, long> bytes = checkpoint.Weights.BytesByClass;
        long dense = bytes[DeepSeekV41WeightClass.Dense] + bytes[DeepSeekV41WeightClass.Embed] + bytes[DeepSeekV41WeightClass.Head];

        ResidencyDemand demand = DeepSeekV41HostPlanner.Demand(checkpoint);

        Assert.Equal(DeepSeekV41WorkingMemory.AnonymousBytes(checkpoint.Config, dense, HfTextDirectoryLoader.LoadOptions), demand.WorkingSet.Values.Sum());
        Assert.Equal(DeepSeekV41HostPlanner.HostHeadroomBytes, demand.HeadroomBytes);
    }
}
