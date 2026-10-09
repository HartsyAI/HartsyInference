using HartsyInference.Engine;
using HartsyInference.Engine.Services;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Deployments through <see cref="TextService"/> on the small V4.1 fixture checkpoint (CPU): the record each deploy completes, and what it reports.</summary>
[Collection(TextServiceSlotsCollection.Name)]
public sealed class TextServiceDeploymentTests : IDisposable
{
    private readonly string _root = Directory.CreateTempSubdirectory("text-deploy-").FullName;

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private static DeploymentRequest Deploy(string id, string path) => new() { DeploymentId = id, Model = TextServiceLeaseTests.Spec(path), Device = "cpu" };

    [Fact]
    public async Task Overlapping_Deploys_Of_One_Id_Do_Not_Throw_And_The_Later_One_Completes_The_Record()
    {
        string path = TextServiceLeaseTests.WriteCheckpoint(_root, "model");
        using InferenceEngine engine = new("cpu", 0);
        ITextService text = engine.Text;

        // The second starts while the first is still loading: the first yields at its load, and nothing awaits it in between.
        Task<DeploymentStatus> first = text.DeployAsync(Deploy("chat", path));
        Task<DeploymentStatus> second = text.DeployAsync(Deploy("chat", path));
        DeploymentStatus[] statuses = await Task.WhenAll(first, second).WaitAsync(TimeSpan.FromSeconds(60));

        // The overtaken first deploy reports the record as it then stood; the second completes it.
        Assert.Contains(statuses[0].State, new[] { DeploymentState.Loading, DeploymentState.Ready });
        Assert.Equal(DeploymentState.Ready, statuses[1].State);
        Assert.Equal(DeploymentState.Ready, Assert.Single(text.Deployments).State);
    }
}
