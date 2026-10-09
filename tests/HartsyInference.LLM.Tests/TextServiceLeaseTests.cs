using HartsyInference.Core.Configuration;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.LLM.Tests.DeepSeekV41;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Slot leases through <see cref="TextService"/> on the small V4.1 fixture checkpoint (CPU): a load that replaces a model waits for the scheduled requests
/// still running on it, decided by the same rule the loader reloads by, and an unload that gives up on them leaves the model taking requests. A test holds a lease the
/// way a running scheduled request does.</summary>
[Collection(TextServiceSlotsCollection.Name)]
public sealed class TextServiceLeaseTests : IDisposable
{
    private readonly string _root = Directory.CreateTempSubdirectory("text-lease-").FullName;

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private string Checkpoint(string name)
    {
        string directory = Directory.CreateDirectory(Path.Combine(_root, name)).FullName;
        DeepSeekV41ModelFixtureCheckpoint.Write(directory);
        DeepSeekV41ModelFixtureCheckpoint.WriteTokenizer(directory);
        return directory;
    }

    private static ModelSpec Spec(string path) => new() { Requested = "dsv41-fixture", Modality = Modality.Text, LocalPath = path };

    private static TextRequest Request() => new()
    {
        Messages = [new TextMessage { Role = TextRole.User, Content = "hi" }],
        MaxTokens = 2,
        Greedy = true,
        Device = "cpu",
    };

    [Fact]
    public async Task A_Load_Of_A_Path_Differing_Only_In_Case_Waits_For_The_Leases_Then_Replaces_The_Model()
    {
        string upper = Checkpoint("Model");
        // The same directory on a case-insensitive filesystem; the load still replaces the model, since the paths differ.
        string lower = Checkpoint("model");
        using InferenceEngine engine = new("cpu", 0);
        TextService text = (TextService)engine.Text;
        await text.GenerateAsync(Spec(upper), Request());
        TextDeviceSlot slot = text.SlotFor("cpu")!;

        slot.EnterLease();
        Task<TextResult> replacing;
        try
        {
            replacing = text.GenerateAsync(Spec(lower), Request());
            await Task.Delay(200);
            Assert.False(replacing.IsCompleted, "the load did not wait for the lease");
        }
        finally
        {
            slot.ExitLease();
        }

        await replacing.WaitAsync(TimeSpan.FromSeconds(60));
        Assert.Equal(lower, slot.LoadedPath);
    }

    [Fact]
    public async Task After_An_Unload_Gives_Up_On_A_Lease_The_Resident_Model_Takes_Scheduled_Requests_Again()
    {
        string path = Checkpoint("model");
        bool overridden = KnobStore.HasOverride(EngineKnobs.ContinuousBatching);
        bool previous = EngineKnobs.ContinuousBatching.Value;
        KnobStore.Set(EngineKnobs.ContinuousBatching, true);
        try
        {
            using InferenceEngine engine = new("cpu", 0);
            TextService text = (TextService)engine.Text;
            text.UnloadLeaseWait = TimeSpan.FromMilliseconds(50);
            await text.GenerateAsync(Spec(path), Request());
            TextDeviceSlot slot = text.SlotFor("cpu")!;
            Assert.NotNull(slot.Scheduler);

            // The unload drains the scheduler, then gives up on the lease, so the model stays resident.
            slot.EnterLease();
            try
            {
                Assert.False(text.Unload("cpu"));
            }
            finally
            {
                slot.ExitLease();
            }

            // A scheduler left draining refused this with SchedulerStoppedException.
            await text.GenerateAsync(Spec(path), Request()).WaitAsync(TimeSpan.FromSeconds(60));
            Assert.Equal(path, slot.LoadedPath);
        }
        finally
        {
            if (overridden)
                KnobStore.Set(EngineKnobs.ContinuousBatching, previous);
            else
                KnobStore.Clear(EngineKnobs.ContinuousBatching);
        }
    }
}
