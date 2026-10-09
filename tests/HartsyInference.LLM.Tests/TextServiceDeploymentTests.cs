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

    [Fact]
    public async Task A_Deploy_Waiting_For_The_Device_Leaves_The_Old_Deployment_Ready_And_Fails_When_Cancelled()
    {
        string path = TextServiceLeaseTests.WriteCheckpoint(_root, "model");
        using InferenceEngine engine = new("cpu", 0);
        TextService text = (TextService)engine.Text;
        await text.DeployAsync(Deploy("old", path));
        TextDeviceSlot slot = text.SlotFor("cpu")!;

        // Something else holds the device, so the new deploy cannot take it.
        await slot.Lock.WaitAsync();
        try
        {
            using CancellationTokenSource cancel = new();
            Task<DeploymentStatus> waiting = text.DeployAsync(Deploy("new", path), cancel.Token);
            // The old deployment still serves while the new one waits, so it still reads Ready.
            Assert.Equal(DeploymentState.Ready, text.Deployments.Single(d => d.DeploymentId == "old").State);
            cancel.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => waiting).WaitAsync(TimeSpan.FromSeconds(30));
        }
        finally
        {
            slot.Lock.Release();
        }

        Assert.Equal(DeploymentState.Ready, text.Deployments.Single(d => d.DeploymentId == "old").State);
        Assert.Equal(DeploymentState.Failed, text.Deployments.Single(d => d.DeploymentId == "new").State);
    }

    [Fact]
    public async Task After_An_Unload_Capacity_And_Deployments_Both_Report_Unloaded()
    {
        string path = TextServiceLeaseTests.WriteCheckpoint(_root, "model");
        using InferenceEngine engine = new("cpu", 0);
        ITextService text = engine.Text;
        Assert.Equal(DeploymentState.Ready, (await text.DeployAsync(Deploy("chat", path))).State);

        Assert.True(text.Unload("cpu"));

        // Capacity first: it read Ready here until a read of Deployments retired the record as a side effect.
        Assert.Equal(DeploymentState.Unloaded, text.Capacity("chat")!.State);
        Assert.Equal(DeploymentState.Unloaded, Assert.Single(text.Deployments).State);
    }

    [Fact]
    public async Task A_Request_That_Loads_Another_Model_On_The_Device_Unloads_Its_Deployment()
    {
        string deployed = TextServiceLeaseTests.WriteCheckpoint(_root, "deployed");
        string other = TextServiceLeaseTests.WriteCheckpoint(_root, "other");
        using InferenceEngine engine = new("cpu", 0);
        ITextService text = engine.Text;
        await text.DeployAsync(Deploy("chat", deployed));

        await text.GenerateAsync(TextServiceLeaseTests.Spec(other), TextServiceLeaseTests.Request());

        Assert.Equal(DeploymentState.Unloaded, text.Capacity("chat")!.State);
        Assert.Equal(DeploymentState.Unloaded, Assert.Single(text.Deployments).State);
    }
}
