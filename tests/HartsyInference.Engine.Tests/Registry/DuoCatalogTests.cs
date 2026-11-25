using HartsyInference.Engine.Registry;

namespace HartsyInference.Engine.Tests.Registry;

public sealed class DuoCatalogTests
{
    private static readonly string[] Ids = ["duo-v1-distill", "duo-v1-causal", "duo-v1-2-causal"];

    [Fact]
    public void EveryDuoReleaseIsCataloguedAsValidationPendingAndNotYetRunnable()
    {
        foreach (string id in Ids)
        {
            CatalogEntry entry = ModelCatalog.All.Single(e => e.Id == id);
            Assert.Equal(ModelStatus.ValidationPending, entry.Status);
            Assert.False(entry.CliDrivable);
            Assert.StartsWith("inclusionAI/Duo-", entry.HuggingFaceRepo, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void EachReleaseListsItsTwoLoraFilesWithPinnedHashesInsideItsOwnFolder()
    {
        foreach (string id in Ids)
        {
            CatalogEntry entry = ModelCatalog.All.Single(e => e.Id == id);
            Assert.Equal(["realization/dcp/__0_0.distcp", "realization/dcp/.metadata"], entry.Assets.Select(a => a.RepoPath));
            Assert.All(entry.Assets, a =>
            {
                Assert.Equal(64, a.Sha256!.Length);
                Assert.All(a.Sha256, c => Assert.True(char.IsAsciiHexDigitLower(c)));
                Assert.Equal(entry.HuggingFaceRepo, a.Repo);
                Assert.StartsWith("Stable-Diffusion/Duo/duo-", a.TargetSubdir, StringComparison.Ordinal);
            });
        }
    }

    [Fact]
    public void TheThreeReleasesCarryDifferentLoraWeightsAndTheCausalPairShareTheirIndex()
    {
        Dictionary<string, string> weights = Ids.ToDictionary(id => id,
            id => ModelCatalog.All.Single(e => e.Id == id).Assets.Single(a => a.RepoPath.EndsWith(".distcp", StringComparison.Ordinal)).Sha256!);
        Assert.Equal(3, weights.Values.Distinct().Count());
        string Index(string id) => ModelCatalog.All.Single(e => e.Id == id).Assets.Single(a => a.RepoPath.EndsWith(".metadata", StringComparison.Ordinal)).Sha256!;
        Assert.Equal(Index("duo-v1-causal"), Index("duo-v1-2-causal"));
        Assert.NotEqual(Index("duo-v1-distill"), Index("duo-v1-causal"));
    }
}
