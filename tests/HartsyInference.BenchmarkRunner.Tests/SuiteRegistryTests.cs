using HartsyInference.BenchmarkRunner.Contracts;
using HartsyInference.BenchmarkRunner.Execution;
using HartsyInference.BenchmarkRunner.Serialization;
using Xunit;

namespace HartsyInference.BenchmarkRunner.Tests;

/// <summary>Workload registration is declarative while submission paths remain outside the trusted registry.</summary>
public sealed class SuiteRegistryTests
{
    [Fact]
    public void NewBundledManifestIsDiscoveredWithoutEditingRunnerDispatch()
    {
        string id = "test-" + Guid.NewGuid().ToString("N") + "-v1";
        string file = Path.Combine(AppContext.BaseDirectory, "suites", id + ".json");
        SuiteDefinition suite = Suites.Load("quick-v1") with { Id = id };
        try
        {
            BenchJson.Write(file, suite, BenchJson.Default.SuiteDefinition);
            Assert.Contains(id, Suites.AvailableIds);
            Assert.Equal(id, Suites.Load(id).Id);
            Assert.Throws<ArgumentException>(() => Suites.PathFor("../submission"));
        }
        finally { File.Delete(file); }
    }
}
