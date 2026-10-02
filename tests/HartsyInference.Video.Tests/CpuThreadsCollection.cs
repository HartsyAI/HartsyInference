using Xunit;

namespace HartsyInference.Video.Tests;

/// <summary>Serializes the test classes that change the process-wide <c>numerics.cpuThreads</c> cap and runs them apart
/// from every other class, so no other test runs under a cap one of them set.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class CpuThreadsCollection
{
    public const string Name = "cpu-threads";
}
