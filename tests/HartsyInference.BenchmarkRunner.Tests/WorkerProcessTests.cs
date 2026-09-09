using HartsyInference.BenchmarkRunner.Contracts;
using HartsyInference.BenchmarkRunner.Execution;
using HartsyInference.BenchmarkRunner.Serialization;
using Xunit;

namespace HartsyInference.BenchmarkRunner.Tests;
/// <summary>Native failures stay in child processes and cannot masquerade as completed sessions.</summary>
public sealed class WorkerProcessTests
{
    [Fact]
    public async Task MissingAssetProducesDurableFailedJournal()
    {
        string root = Path.Combine(Path.GetTempPath(), "hartsy-worker-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            string caseId = Suites.Load("quick-v1").Cases[0].Id;
            int code = await ChildProcess.RunAsync(["worker", root, root, "quick-v1", caseId, "cpu", "0", "1"], Path.Combine(root,
                "worker.log"), TimeSpan.FromSeconds(30), CancellationToken.None);
            Assert.Equal(1, code);
            SessionRecord session = BenchJson.Read(Path.Combine(root, "sessions", caseId, "0", "1", "session.json"), BenchJson.Default
                .SessionRecord);
            Assert.Equal("failed", session.Status);
            Assert.Empty(session.Measurements);
            Assert.Equal("InvalidDataException", session.Failure);
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    [Fact]
    public async Task DeadlineKillsChildAndReturnsTimeout()
    {
        string root = Path.Combine(Path.GetTempPath(), "hartsy-timeout-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            int code = await ChildProcess.RunAsync(["probe", "cpu", Path.Combine(root, "device.json")], Path.Combine(root, "probe.log"),
                TimeSpan.FromMilliseconds(1), CancellationToken.None);
            Assert.Equal(124, code);
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }
}
