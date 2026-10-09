using HartsyInference.Core.Configuration;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
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
    public async Task Unloading_A_Replaced_Deployment_Leaves_The_One_That_Replaced_It_Resident()
    {
        string pathA = TextServiceLeaseTests.WriteCheckpoint(_root, "a");
        string pathB = TextServiceLeaseTests.WriteCheckpoint(_root, "b");
        using InferenceEngine engine = new("cpu", 0);
        TextService text = (TextService)engine.Text;
        text.UnloadLeaseWait = TimeSpan.FromMilliseconds(50);
        await text.DeployAsync(Deploy("a", pathA));
        await text.DeployAsync(Deploy("b", pathB));
        TextDeviceSlot slot = text.SlotFor("cpu")!;

        // b replaced a on the device: unloading a must leave b's model where it is.
        Assert.Equal(DeploymentUnloadOutcome.AlreadyGone, text.UnloadDeployment("a"));
        Assert.Equal(pathB, slot.LoadedPath);
        Assert.Equal(DeploymentState.Ready, text.Deployments.Single(d => d.DeploymentId == "b").State);

        // A scheduled request still running on b: the unload gives up, and b keeps serving.
        slot.EnterLease();
        try
        {
            Assert.Equal(DeploymentUnloadOutcome.TimedOut, text.UnloadDeployment("b"));
        }
        finally
        {
            slot.ExitLease();
        }
        Assert.Equal(pathB, slot.LoadedPath);

        Assert.Equal(DeploymentUnloadOutcome.Unloaded, text.UnloadDeployment("b"));
        Assert.Null(slot.LoadedPath);
        Assert.Equal(DeploymentState.Unloaded, text.Deployments.Single(d => d.DeploymentId == "b").State);
        Assert.Equal(DeploymentUnloadOutcome.NotFound, text.UnloadDeployment("missing"));
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

    private static DeploymentRequest DeployNamed(string id, string requested, string path) =>
        new() { DeploymentId = id, Model = new ModelSpec { Requested = requested, Modality = Modality.Text, LocalPath = path }, Device = "cpu" };

    [Fact]
    public async Task A_Redeploy_Of_A_Serving_Deployment_Reads_Ready_While_It_Waits_And_Ready_On_The_New_Model_After()
    {
        string pathA = TextServiceLeaseTests.WriteCheckpoint(_root, "a");
        string pathB = TextServiceLeaseTests.WriteCheckpoint(_root, "b");
        using InferenceEngine engine = new("cpu", 0);
        TextService text = (TextService)engine.Text;
        await text.DeployAsync(DeployNamed("chat", "model-a", pathA));
        TextDeviceSlot slot = text.SlotFor("cpu")!;
        text.UnloadLeaseWait = TimeSpan.FromSeconds(60);

        // A scheduled request still runs on model a, so the redeploy waits for it before it frees a.
        slot.EnterLease();
        Task<DeploymentStatus> redeploy;
        try
        {
            redeploy = text.DeployAsync(DeployNamed("chat", "model-b", pathB));
            // Nothing is freed yet, so the deployment still reads as serving model a.
            DeploymentStatus waiting = Assert.Single(text.Deployments);
            Assert.Equal(("model-a", DeploymentState.Ready), (waiting.Model, waiting.State));
            await Task.Delay(100);
            Assert.False(redeploy.IsCompleted, "the redeploy did not wait for the lease");
            Assert.Equal(DeploymentState.Ready, Assert.Single(text.Deployments).State);
        }
        finally
        {
            slot.ExitLease();
        }

        Assert.Equal(DeploymentState.Ready, (await redeploy.WaitAsync(TimeSpan.FromSeconds(60))).State);
        DeploymentStatus record = Assert.Single(text.Deployments);
        Assert.Equal(("model-b", DeploymentState.Ready), (record.Model, record.State));
        Assert.Equal(pathB, slot.LoadedPath);
    }

    [Fact]
    public async Task A_Redeploy_That_Times_Out_On_Its_Lease_Fails_And_Leaves_The_Serving_Deployment_Ready()
    {
        string pathA = TextServiceLeaseTests.WriteCheckpoint(_root, "a");
        string pathB = TextServiceLeaseTests.WriteCheckpoint(_root, "b");
        using InferenceEngine engine = new("cpu", 0);
        TextService text = (TextService)engine.Text;
        await text.DeployAsync(DeployNamed("chat", "model-a", pathA));
        TextDeviceSlot slot = text.SlotFor("cpu")!;
        text.UnloadLeaseWait = TimeSpan.FromMilliseconds(50);

        DeploymentStatus status;
        slot.EnterLease();
        try
        {
            status = await text.DeployAsync(DeployNamed("chat", "model-b", pathB)).WaitAsync(TimeSpan.FromSeconds(60));
        }
        finally
        {
            slot.ExitLease();
        }

        // The redeploy freed nothing: its own status reports the failure, and the record still describes the model that serves.
        Assert.Equal(DeploymentState.Failed, status.State);
        Assert.Contains("did not finish", status.Problem);
        DeploymentStatus record = Assert.Single(text.Deployments);
        Assert.Equal(("model-a", DeploymentState.Ready), (record.Model, record.State));
        Assert.Equal(pathA, slot.LoadedPath);
        Assert.Equal(DeploymentState.Ready, text.Capacity("chat")!.State);
    }

    [Fact]
    public async Task A_Cancelled_Redeploy_Of_A_Serving_Deployment_Leaves_It_Ready()
    {
        string pathA = TextServiceLeaseTests.WriteCheckpoint(_root, "a");
        string pathB = TextServiceLeaseTests.WriteCheckpoint(_root, "b");
        using InferenceEngine engine = new("cpu", 0);
        TextService text = (TextService)engine.Text;
        await text.DeployAsync(DeployNamed("chat", "model-a", pathA));
        TextDeviceSlot slot = text.SlotFor("cpu")!;

        // Something else holds the device, so the redeploy waits for it and is cancelled there.
        await slot.Lock.WaitAsync();
        try
        {
            using CancellationTokenSource cancel = new();
            Task<DeploymentStatus> waiting = text.DeployAsync(DeployNamed("chat", "model-b", pathB), cancel.Token);
            Assert.Equal(DeploymentState.Ready, Assert.Single(text.Deployments).State);
            cancel.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => waiting).WaitAsync(TimeSpan.FromSeconds(30));
        }
        finally
        {
            slot.Lock.Release();
        }

        DeploymentStatus record = Assert.Single(text.Deployments);
        Assert.Equal(("model-a", DeploymentState.Ready), (record.Model, record.State));
        Assert.Equal(pathA, slot.LoadedPath);
    }

    [Fact]
    public async Task A_Redeploy_That_Frees_The_Model_And_Then_Fails_Reads_Failed_With_The_Model_Gone()
    {
        string pathA = TextServiceLeaseTests.WriteCheckpoint(_root, "a");
        // Passes every check before the free, and fails only when the load reads it.
        string broken = Path.Combine(_root, "broken.gguf");
        await File.WriteAllTextAsync(broken, "not a model");
        using InferenceEngine engine = new("cpu", 0);
        TextService text = (TextService)engine.Text;
        await text.DeployAsync(DeployNamed("chat", "model-a", pathA));
        TextDeviceSlot slot = text.SlotFor("cpu")!;

        DeploymentStatus status = await text.DeployAsync(DeployNamed("chat", "model-broken", broken));

        Assert.Equal(DeploymentState.Failed, status.State);
        DeploymentStatus record = Assert.Single(text.Deployments);
        Assert.Equal(("model-broken", DeploymentState.Failed), (record.Model, record.State));
        // The free happened, so the old model is gone and the record must not claim it still serves.
        Assert.Null(slot.LoadedPath);
    }

    [Fact]
    public async Task Capacity_Of_A_Replaced_Deployment_Reports_No_Figures()
    {
        string pathA = TextServiceLeaseTests.WriteCheckpoint(_root, "a");
        string pathB = TextServiceLeaseTests.WriteCheckpoint(_root, "b");
        bool overridden = KnobStore.HasOverride(EngineKnobs.ContinuousBatching);
        bool previous = EngineKnobs.ContinuousBatching.Value;
        KnobStore.Set(EngineKnobs.ContinuousBatching, true);
        try
        {
            using InferenceEngine engine = new("cpu", 0);
            TextService text = (TextService)engine.Text;
            await text.DeployAsync(DeployNamed("a", "model-a", pathA));
            await text.DeployAsync(DeployNamed("b", "model-b", pathB));
            // A scheduled request on model b gives the device a scheduler. Deployment a must not report it.
            await text.GenerateAsync(TextServiceLeaseTests.Spec(pathB), TextServiceLeaseTests.Request());
            Assert.True(text.Capacity("b")!.MaxConcurrent > 0);

            DeploymentCapacity replaced = text.Capacity("a")!;
            Assert.Equal((DeploymentState.Unloaded, 0, 0, 0), (replaced.State, replaced.Active, replaced.Queued, replaced.MaxConcurrent));
            Assert.Null(replaced.KvPagesFree);
            Assert.Null(replaced.KvPagesTotal);
        }
        finally
        {
            if (overridden)
                KnobStore.Set(EngineKnobs.ContinuousBatching, previous);
            else
                KnobStore.Clear(EngineKnobs.ContinuousBatching);
        }
    }

    [Fact]
    public async Task A_Redeploy_That_Keeps_The_Loaded_Model_Reads_Ready_Under_The_New_Name()
    {
        string path = TextServiceLeaseTests.WriteCheckpoint(_root, "model");
        using InferenceEngine engine = new("cpu", 0);
        TextService text = (TextService)engine.Text;
        await text.DeployAsync(DeployNamed("chat", "model-a", path));
        TextDeviceSlot slot = text.SlotFor("cpu")!;

        // The same checkpoint is already loaded, so nothing is freed: the record completes its pending load in place.
        DeploymentStatus status = await text.DeployAsync(DeployNamed("chat", "model-a-renamed", path));

        Assert.Equal(DeploymentState.Ready, status.State);
        DeploymentStatus record = Assert.Single(text.Deployments);
        Assert.Equal(("model-a-renamed", DeploymentState.Ready), (record.Model, record.State));
        Assert.Equal(path, slot.LoadedPath);
    }
}
