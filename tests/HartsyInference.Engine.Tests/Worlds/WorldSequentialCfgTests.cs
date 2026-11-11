using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Worlds;

namespace HartsyInference.Engine.Tests.Worlds;

public sealed class WorldSequentialCfgTests
{
    [Fact]
    public void IsOffByDefault_AndLeavesThePlanSnapshotUnchanged()
    {
        ModelSpec oasis = WorldTestFixtures.Spec(ModelCatalog.Find("oasis"), "oasis");

        WorldPlan plan = WorldPlanResolver.Resolve(oasis, new WorldRequest { InitImage = WorldTestFixtures.Image(640, 360), Seed = 3 }, "cpu");

        Assert.False(plan.SequentialCfg);
        Assert.False(plan.Request.SequentialCfg);
        Assert.DoesNotContain("sequentialCfg", plan.Snapshot());
    }

    [Fact]
    public void IsRefusedForAModelThatDoesNotDeclareIt()
    {
        ModelSpec oasis = WorldTestFixtures.Spec(ModelCatalog.Find("oasis"), "oasis");
        WorldRequest request = new WorldRequest { InitImage = WorldTestFixtures.Image(640, 360), SequentialCfg = true };

        WorldPlanningException ex = Assert.Throws<WorldPlanningException>(() => WorldPlanResolver.Resolve(oasis, request, "cpu"));

        Assert.Contains("no sequential classifier-free guidance path", ex.Message);
    }

    [Fact]
    public void IsCarriedOnThePlan_WhenTheModelDeclaresIt()
    {
        CatalogEntry entry = WorldTestFixtures.FixtureEntry(new WorldCapabilities { SupportsGuidance = true, SupportsSequentialCfg = true });
        WorldRequest request = new WorldRequest { InitImage = WorldTestFixtures.Image(8, 8), Seed = 3, SequentialCfg = true };

        WorldPlan plan = WorldPlanResolver.Resolve(WorldTestFixtures.Spec(entry), request, "cpu");

        Assert.True(plan.SequentialCfg);
        Assert.True(plan.Request.SequentialCfg);
        Assert.Contains("sequentialCfg=True", plan.Snapshot());
    }

    [Fact]
    public void Options_MapTheFlagOntoTheRequest()
    {
        Assert.True(new WorldRequestOptions { SequentialCfg = true }.ToRequest(WorldTestFixtures.Image(8, 8)).SequentialCfg);
        Assert.False(new WorldRequestOptions().ToRequest(WorldTestFixtures.Image(8, 8)).SequentialCfg);
    }
}
